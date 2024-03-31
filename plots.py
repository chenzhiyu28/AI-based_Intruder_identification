from numpy import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.animation as animation
import math
from clustering import *
#from data2 import *
from eeee import *
from matplotlib.patches import Rectangle

# Global limits and starts for x and y axis
X_AXIS_START = -1
Y_AXIS_START = -1
X_AXIS_LIMIT = 3
Y_AXIS_LIMIT = 6


def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
        return data

# data
# data = read_data("data.txt")
# print(data)


# 可视化
class Visualizer:
    def __init__(self, data):
        self.data = data
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = plt.axes([0.1, 0.2, 0.8, 0.75])
        self.current_index = 0
        self.paused = True
        self.animation = animation.FuncAnimation(self.fig, self.update_plot, interval=100, repeat=False,
                                                 save_count=len(data))
        self.time_text = self.ax.text(0.05, 0.02, '', transform=self.ax.transAxes)
        self.target_lost = False
        self.lost_time = 0

        # x,y 轴的 尺度
        self.ax.set_xlim(X_AXIS_START, X_AXIS_LIMIT)
        self.ax.set_ylim(Y_AXIS_START, Y_AXIS_LIMIT)

        # 四个按钮
        self.btn_ax_play = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.btn_play = Button(self.btn_ax_play, 'Play')
        self.btn_play.on_clicked(self.play)

        self.btn_ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])
        self.btn_next = Button(self.btn_ax_next, 'Next')
        self.btn_next.on_clicked(self.next_frame)

        self.btn_ax_pause = plt.axes([0.6, 0.05, 0.1, 0.075])
        self.btn_pause = Button(self.btn_ax_pause, 'Pause')
        self.btn_pause.on_clicked(self.pause)

        self.btn_ax_replay = plt.axes([0.5, 0.05, 0.1, 0.075])
        self.btn_replay = Button(self.btn_ax_replay, 'Replay')
        self.btn_replay.on_clicked(self.replay)

        # 目前只考虑1个 centroid; 分别存储x,和y的坐标
        self.centroid_history = [[], []]

        self.initial_frame()

    # for each cluster, generate a rectangle to contain all points
    def generate_rectangle(self, x_values, y_values):
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        width = x_max - x_min
        height = y_max - y_min
        rectangle = Rectangle((x_min, y_min), width, height, fill=False, color='blue', linewidth=1)
        self.ax.add_patch(rectangle)

    # 这里centroid 耦合死了, 适合创建一个 point class
    def _updating_centroid_history(self, centroid):
        self.centroid_history[0].append(centroid[0])
        self.centroid_history[1].append(centroid[1])

    def clear_centroid_history(self):
        self.centroid_history = [[], []]

    def _get_frame_centroids(self, x_values, y_values, k=1):
        clustering = KmeansClustering(k, x_values, y_values)
        centroids = DensityBasedClustering.get_center()
        return centroids

    def _density_centroid(self, cluster):
        return DensityBasedClustering.get_center(cluster)

    # 得到density cluster 的 [x_val], [y_val], 并画上centroid
    def _density_plot_cluster_points(self, x_values, y_values, min_points, eps):
        d = DensityBasedClustering(x_values, y_values, min_points, eps)

        cluster = d.clustering()
        if cluster:
            cluster = cluster[0]
            x_val, y_val = d.get_x_y_list(cluster)

            # centroid updating
            centroid = self._density_centroid(cluster)
            self._updating_centroid_history(centroid)
            self.ax.scatter(centroid[0], centroid[1], color=(0.5, 0, 0))

            return x_val, y_val
        return None, None

    # 把centroid点更新在图上
    def plot_centroid(self, x_values, y_values, k=1):
        centroids = self._get_frame_centroids(x_values, y_values)
        x, y = [], []
        for centroid in centroids:
            x.append(centroid[0])
            y.append(centroid[1])
            self._updating_centroid_history(centroid)
        self.ax.scatter(x, y, color=(0.5, 0, 0))

    # 在图上展示所有的点Plot the data points and their centroid.
    def plot_points(self, x_vals, y_vals):
        self.ax.scatter(x_vals, y_vals)
        # self.plot_centroid(x_vals, y_vals)

        # density plot
        xs, ys = self._density_plot_cluster_points(x_vals, y_vals, 4, eps=0.3)
        if xs and ys:
            self.ax.scatter(xs, ys, color=(0, 0.5, 0))
            self.generate_rectangle(xs, ys)
        else:
            self.target_lost = True


        # 历史轨迹,线图
        self.ax.plot(self.centroid_history[0], self.centroid_history[1], color=(0, 0, 0.5), alpha=0.3)

    def initial_frame(self):
        self.current_index = 0
        frame = self.data[0]
        self.ax.clear()

        self.time_text = self.ax.text(0.05, 0.02, f'Time: {frame["time"]}', transform=self.ax.transAxes)
        self.ax.set_xlim(X_AXIS_START, X_AXIS_LIMIT)
        self.ax.set_ylim(Y_AXIS_START, Y_AXIS_LIMIT)

        # Plot the data points
        self.plot_points(frame['x'], frame['y'])

        plt.draw()

    def update_plot(self, _):
        if not self.paused and self.current_index < len(self.data):
            frame = self.data[self.current_index]
            self.ax.clear()
            self.time_text = self.ax.text(0.05, 0.02, f'Time: {frame["time"]}', transform=self.ax.transAxes)
            self.ax.set_xlim(X_AXIS_START, X_AXIS_LIMIT)
            self.ax.set_ylim(Y_AXIS_START, Y_AXIS_LIMIT)

            # Plot the data points
            self.plot_points(frame['x'], frame['y'])

            self.current_index += 1

    def next_frame(self, event):
        if self.paused and self.current_index < len(self.data) - 1:
            self.current_index += 1
            frame = self.data[self.current_index]
            self.ax.clear()
            self.time_text = self.ax.text(0.05, 0.02, f'Time: {frame["time"]}', transform=self.ax.transAxes)
            self.ax.set_xlim(X_AXIS_START, X_AXIS_LIMIT)
            self.ax.set_ylim(Y_AXIS_START, Y_AXIS_LIMIT)

            # Plot the data points
            self.plot_points(frame['x'], frame['y'])

            plt.draw()

    def pause(self, event):
        self.paused = True

    def play(self, event):
        if self.paused:
            self.paused = False

    def replay(self, event):
        self.paused = True
        self.clear_centroid_history()
        self.initial_frame()

data = FRAMES
viz = Visualizer(data)
plt.show()