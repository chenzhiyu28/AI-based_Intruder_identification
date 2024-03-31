import pprint
import matplotlib.pyplot as plt
import numpy as np
from clustering import *
# from data2 import *
from circle_data import *
from csaps import csaps
import statistics


class Preprocessing:
    def __init__(self, frame_data):
        self.data = frame_data
        self.velocity_filter = True
        self.velocity_filter_threshold = 6.5

        # matplotlib的尺寸 plot size
        self.x_start, self.x_end = -1.5, 1.5
        self.y_start, self.y_end = -0.5, 5
        self.SMOOTH_RATE = 0.8

        # 按时间顺序的 centroids, x,y,time
        self.CENTROIDS = []
        self.x_values = []  # 没啥意义
        self.y_values = []  # 没啥意义
        self.TIME = []

        # 持续时间
        self.LAST_TIME = round(self.data[-1]["time"], 2)
        self.MISSING_TIMESTAMP = []

        # 记录有多少数据被 velocity filter 修正过
        self.filtered_record = 0
        self.filtered_record_all = 0

        # 按照时间排序的速度
        self.SPEED = []  # 这里的speed 只在有centroid时计算 (一个人跑到座位坐下后, 一直坐着读不到数据,那么坐着开始后就不计算了)
        self.SPEED_SMOOTH = []
        self.speed_variance = None

        self.velocity = []
        self.velocity_smooth = []
        self.velocity_variance = None

        self.x_smooth = []
        self.y_smooth = []
        self.speed_variance_smoothed = None
        self.velocity_variance_smoothed = None

        # for approaching tendency
        self.group = []
        self.all_centroids = []
        self.all_timestamp = []
        self.all_centroids_after_imputation = []
        self.all_speed = []  # 必须设置filter为 True
        self.stillness_percentage = None
        self.approaching_tendency = None
        self.suspicious_time_percent = None

        # statistics
        self.average_speed = None
        self.average_velocity_module = None
        self.speed_variance = None
        self.velocity_variance = None

    @staticmethod
    def calculate_speed(coordinate1, coordinate2, time1, time2):
        x1, y1 = coordinate1[0], coordinate1[1]
        x2, y2 = coordinate2[0], coordinate2[1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        speed = distance / (time2 - time1)
        return round(speed, 2)

    # 用smoothed之后的speed, 计算variance
    def calculate_speed_variance_smoothed(self, x_values, y_values, timestamps):
        for i in range(len(x_values) - 1):
            x1, x2 = x_values[i], x_values[i + 1]
            y1, y2 = y_values[i], y_values[i + 1]
            time1, time2 = timestamps[i], timestamps[i + 1]

            # 计算速度
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            speed = distance / (time2 - time1)

            self.SPEED_SMOOTH.append(round(speed, 2))

        # 计算方差
        if len(self.SPEED_SMOOTH) >= 2:
            variance = statistics.variance(self.SPEED_SMOOTH)
        else:
            variance = 0
        return variance

    # 直接用原始数据计算方差, 可以不写这个函数的
    def calculate_speed_variance(self):
        for i in range(len(self.CENTROIDS) - 1):
            speed = self.calculate_speed(self.CENTROIDS[i], self.CENTROIDS[i + 1], self.TIME[i], self.TIME[i + 1])
            self.SPEED.append(speed)

        if len(self.SPEED) >= 2:
            variance = statistics.variance(self.SPEED)
        else:
            variance = 0
        return variance

    # 返回(v_x, v_y) 带正负号速度
    @staticmethod
    def calculate_velocity(coordinate1, coordinate2, time1, time2) -> tuple[float, float]:
        x1, y1 = coordinate1[0], coordinate1[1]
        x2, y2 = coordinate2[0], coordinate2[1]
        distance_x = x2 - x1
        distance_y = y2 - y1
        time = time2 - time1

        speed_x = round((distance_x / time), 2)
        speed_y = round((distance_y / time), 2)
        velocity = speed_x, speed_y

        return velocity

    def calculate_velocity_variance(self):
        # 得到原始数据的velocity
        for i in range(len(self.CENTROIDS) - 1):
            velocity = self.calculate_velocity(self.CENTROIDS[i], self.CENTROIDS[i + 1], self.TIME[i], self.TIME[i + 1])
            self.velocity.append(velocity)

        velocity_x, velocity_y = [], []

        # x,y分速度分别计算variance, 再算整体的variance
        for velocity_ in self.velocity:
            velocity_x.append(velocity_[0])
            velocity_y.append(velocity_[1])

        if len(velocity_x) >= 2:
            variance_x = statistics.variance(velocity_x)
            variance_y = statistics.variance(velocity_y)

        else:
            variance_x, variance_y = 0, 0

        variance = round((variance_x ** 2 + variance_y ** 2) ** 0.5, 2)
        return variance

    # 用smoothed之后的speed, 计算variance
    def calculate_velocity_variance_smoothed(self, x_values, y_values, timestamps):
        for i in range(len(x_values) - 1):
            x1, x2 = x_values[i], x_values[i + 1]
            y1, y2 = y_values[i], y_values[i + 1]
            time1, time2 = timestamps[i], timestamps[i + 1]

            # 计算速度
            velocity = self.calculate_velocity((x1, y1), (x2, y2), time1, time2)

            self.velocity_smooth.append(velocity)

        velocity_x, velocity_y = [], []

        for velocity_ in self.velocity_smooth:
            velocity_x.append(velocity_[0])
            velocity_y.append(velocity_[1])

        if len(velocity_x) >= 2:
            variance_x = statistics.variance(velocity_x)
            variance_y = statistics.variance(velocity_y)
        else:
            variance_x, variance_y = 0, 0

        variance = round((variance_x ** 2 + variance_y ** 2) ** 0.5, 2)
        return variance

    def p2p_distance(self, point1, point2=(0, 0)) -> float:
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]

        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return round(distance, 3)

    # return the closest distance (in a list of centroids)
    def closest_distance(self, centroids) -> float:
        closest_distance = float(inf)

        for centroid in centroids:
            if centroid and (self.p2p_distance(centroid) < closest_distance):
                closest_distance = self.p2p_distance(centroid)

        if closest_distance == float(inf):
            closest_distance = None

        return closest_distance

    # 返回需要被插补的数据, 分别是 开始,开始index, 结束,结束index
    def get_to_be_imputed_frames(self) -> list[tuple, int, tuple, int]:
        last_centroid = None
        next_centroid = None
        frame_gap = None

        first_centroid_index = None  # 不会用到,方便逻辑顺畅
        first_imputable_index = None
        last_valid_centroid_index = None

        none_indexes = []
        to_be_imputed = []

        index_count = 0  # 找到none centroid的index, 后面不会用了
        for centroid in self.all_centroids:
            if centroid is None:
                none_indexes.append(index_count)
            index_count += 1

        none_pointer = 0

        # 找到第一个 可以被插补的none值
        for i in none_indexes:
            if none_pointer not in none_indexes:
                first_centroid_index = none_pointer  # 不会用到,方便逻辑顺畅
                first_imputable_index = none_indexes[none_pointer]
                break
            none_pointer += 1

        # 找到最后一个 valid centroid (可以被插补的index 都在这个之前)
        last_valid_centroid_index = len(self.all_centroids) - 1
        while True:
            if last_valid_centroid_index in none_indexes:
                last_valid_centroid_index -= 1
            else:
                break

        # 找到下一个不是none的centroid
        start_index = first_imputable_index

        for beginning_index in none_indexes:

            last_centroid = self.all_centroids[start_index - 1]  # none centroid 之前的1个 valid centroid

            if beginning_index < start_index:
                beginning_index = start_index

            if beginning_index >= last_valid_centroid_index:
                break

            # 找到next centroid
            while True:
                beginning_index += 1
                if beginning_index not in none_indexes:
                    next_centroid = self.all_centroids[beginning_index]

                    # 写入需要插补的位置
                    impute = (last_centroid, start_index - 1, next_centroid, beginning_index)
                    to_be_imputed.append(impute)

                    next_index = [x for x in none_indexes if x > beginning_index]
                    if len(next_index) > 0:
                        start_index = min([x for x in none_indexes if x > beginning_index])
                    break

                if beginning_index > last_valid_centroid_index:
                    break

        """
        print("")
        print(f"first imputable index: {first_imputable_index}")
        print(f"last valid centroid index: {last_valid_centroid_index}")
        print(f"to be imputed: {to_be_imputed}")
        """

        return to_be_imputed

    def impute(self) -> None:
        to_be_imputed_frames = self.get_to_be_imputed_frames()
        self.all_centroids_after_imputation = self.all_centroids.copy()

        # print(f"to_be_imputed_frames: {to_be_imputed_frames}")

        for record in to_be_imputed_frames:
            x1, y1 = record[0][0], record[0][1]
            x2, y2 = record[2][0], record[2][1]
            start_frame, end_frame = record[1], record[3]

            total_time = self.all_timestamp[end_frame] - self.all_timestamp[start_frame]
            start_time_point = self.all_timestamp[start_frame]
            delta_x = x2 - x1
            delta_y = y2 - y1

            for i in range(start_frame + 1, end_frame):
                movement_complete_percent = (self.all_timestamp[i] - start_time_point) / total_time
                self.all_centroids_after_imputation[i] = (
                round(x1 + movement_complete_percent * delta_x, 3), round(y1 + movement_complete_percent * delta_y, 3))

            """
            # 用来对 all 数据来插补
            apply_filter_index_for_all = []
            
            for i in range(len(self.all_centroids) - 1):
                centroid, next_centroid = self.all_centroids[i], self.all_centroids[i + 1]
                time1, time2 = self.all_timestamp[i], self.all_timestamp[i+1]

                velocity = self.calculate_velocity(centroid, next_centroid, time1, time2)
                velocity_module = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5

                if velocity_module > 7:
                    apply_filter_index_for_all.append(i)
            """

    # 是否是靠近趋势?
    def calculate_approaching_tendency(self):
        close_trend = 0

        """
        divide the Frames into 10/5 groups
        find the closest distance of each group
        see if it has a tendency to approach
        """
        distances = []
        centroids_in_n_groups = []

        # 把时间段分成5/10份
        if self.LAST_TIME > 10:
            n = 10
        else:
            n = 5

        sample_number = int(len(self.all_centroids) / n)

        while len(centroids_in_n_groups) < n:
            centroids = []
            for i in range(sample_number * len(centroids_in_n_groups),
                           sample_number * len(centroids_in_n_groups) + sample_number):
                centroids.append(self.all_centroids_after_imputation[i])
            centroids_in_n_groups.append(centroids)

        for centroids in centroids_in_n_groups:
            distances.append(self.closest_distance(centroids))

        last_closest_distance = float(Inf)

        for distance in distances:
            if distance:
                last_closest_distance = distance
                break

        for distance in distances:
            if distance:
                if distance < 0.95 * last_closest_distance:
                    close_trend += 1
                last_closest_distance = distance

        self.approaching_tendency = round((close_trend / n), 2)

        """
        print(len(centroids_in_n_groups))
        print(centroids_in_n_groups)
        print(distances)
        print(f"approaching tendency: {self.approaching_tendency}")
        """

    # 一定要在插补完成后 在执行;   开始和结尾的none值会影响数字大小
    def calculate_time_within_suspicious_range(self):
        distances = []
        suspicious_count = 0
        number_of_none_at_beginning = 0
        number_of_none_at_end = 0
        last_position = None

        # 计算开头的none 数量
        for centroid in self.all_centroids_after_imputation:
            if not centroid:
                number_of_none_at_beginning += 1
            else:
                break

        for centroid in self.all_centroids_after_imputation:
            if centroid:
                distances.append(self.p2p_distance(centroid))

        for distance in distances:
            if distance <= 0.5:
                suspicious_count += 1

        all_centroids_copy = self.all_centroids_after_imputation.copy()
        all_centroids_copy.reverse()

        # 计算结尾 none的数量
        for centroid in all_centroids_copy:
            if not centroid:
                number_of_none_at_end += 1
            else:
                last_position = centroid
                break

        if self.p2p_distance(last_position) <= 0.5:
            all_time_count = len(
                self.all_centroids_after_imputation) - number_of_none_at_end - number_of_none_at_beginning
        else:
            all_time_count = len(self.all_centroids_after_imputation) - number_of_none_at_beginning

        self.suspicious_time_percent = round((suspicious_count / all_time_count), 3)

    # 对于无cluster的帧, self.Centroids 是不记录的
    def update_all_centroids(self):
        for frame in self.data:
            self.x_values, self.y_values = frame['x'], frame['y']  # x,y 的array
            cluster = DensityBasedClustering(self.x_values, self.y_values, min_points=4, eps=0.3).clustering()

            if cluster:
                cluster = cluster[0]
                centroid = DensityBasedClustering.get_center(cluster)
                self.CENTROIDS.append(centroid)
                self.TIME.append(round(frame['time'], 2))
                # for approaching tendency
                self.all_centroids.append(centroid)
            else:
                self.MISSING_TIMESTAMP.append(round(frame["time"], 2))
                # for approaching tendency
                self.all_centroids.append(None)

            self.all_timestamp.append(frame['time'])

        # 启用速度 filter
        if self.velocity_filter:
            apply_filter_index = []

            # 找出有问题的centroid index, 对应的 time index 也是这个(也许考虑加到missing timestamp 里)
            for i in range(len(self.CENTROIDS) - 1):
                centroid, next_centroid = self.CENTROIDS[i], self.CENTROIDS[i + 1]
                time1, time2 = self.TIME[i], self.TIME[i + 1]

                velocity = self.calculate_velocity(centroid, next_centroid, time1, time2)
                velocity_module = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5

                if velocity_module >= self.velocity_filter_threshold or velocity_module <= 0 - self.velocity_filter_threshold:
                    self.filtered_record += 1
                    apply_filter_index.append(i)

            apply_filter_index_for_all = []

            all_velocity_modules = []

            # 倒序遍历all centroid, 会得到倒过来的结果
            for i in range(len(self.all_centroids) - 1, -1, -1):
                if self.all_centroids[i] is None:
                    all_velocity_modules.append(None)
                else:
                    next_centroid = self.all_centroids[i]
                    time2 = self.all_timestamp[i]
                    last_index = i
                    while True:
                        i -= 1

                        if i < 0:
                            all_velocity_modules.append(None)
                            break

                        if self.all_centroids[i] is not None:
                            centroid = self.all_centroids[i]
                            time1 = self.all_timestamp[i]

                            velocity = self.calculate_velocity(centroid, next_centroid, time1, time2)
                            velocity_module = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5
                            velocity_module = round(velocity_module, 2)
                            all_velocity_modules.append(velocity_module)
                            break
            all_velocity_modules.reverse()

            for i in range(len(all_velocity_modules)):
                velocity = all_velocity_modules[i]
                if (velocity is not None) and abs(velocity) >= self.velocity_filter_threshold:
                    apply_filter_index_for_all.append(i)

            # 找到有问题的 数据后, 设置对应位置的数据为none(对于all)
            for i in apply_filter_index_for_all:
                self.filtered_record_all += 1
                self.all_centroids[i] = None

            # 删掉错误的数据(对于基础)
            apply_filter_index.reverse()
            # print(apply_filter_index)
            for i in apply_filter_index:
                del self.CENTROIDS[i]
                self.MISSING_TIMESTAMP.append(self.TIME[i])
                del self.TIME[i]

            """
            print(len(self.all_centroids))
            print(len(all_velocity_modules))
            print(self.all_centroids)
            print(all_velocity_modules)
            print(f"找到有问题的: {apply_filter_index}")
            print(f"找到有问题的所有: {apply_filter_index_for_all}")
            """

    def coordinates_to_xy(self, coordinates):
        x, y = [], []
        for coordinate in coordinates:
            x.append(coordinate[0])
            y.append(coordinate[1])
        return np.array(x), np.array(y)

    def generate_statistics(self):
        imputed_centroids = self.all_centroids_after_imputation.copy()

        # find 1st not null value and last not null value
        start, end = 0, 0
        speeds = []
        velocity_modules = []

        for i in range(len(imputed_centroids)):
            centroid = imputed_centroids[i]
            if centroid is not None:
                start = i
                break

        imputed_centroids.reverse()

        for i in range(len(imputed_centroids)):
            centroid = imputed_centroids[i]
            if centroid is not None:
                end = i
                break

        if end != 0:
            centroids = self.all_centroids_after_imputation[start: 0 - end]
            time = self.all_timestamp[start: 0 - end]
        else:
            centroids = self.all_centroids_after_imputation[start:]
            time = self.all_timestamp[start:]

        for i in range(1, len(centroids)):
            distance = self.p2p_distance(centroids[i - 1], centroids[i])
            delta_t = time[i] - time[i - 1]
            speed = distance / delta_t
            speeds.append(round(speed, 2))

            velocity = self.calculate_velocity(centroids[i - 1], centroids[i], time[i - 1], time[i])
            velocity_module = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5
            velocity_modules.append(round(velocity_module, 2))

        # 对于缺失的值, 全部补上0
        speed_error = len(self.all_centroids) - len(speeds)
        for i in range(speed_error):
            speeds.append(0)
            velocity_modules.append(0)

        self.average_speed = round(average(speeds), 2)
        self.average_velocity_module = round(average(velocity_modules), 2)
        self.speed_variance = round(statistics.variance(speeds), 2)
        self.velocity_variance = round(statistics.variance(velocity_modules), 2)

    def show_plot(self):
        plt.figure(figsize=(8, 6))
        ax = plt.axes([0.1, 0.2, 0.8, 0.75])
        ax.set_xlim(self.x_start, self.x_end)
        ax.set_ylim(self.y_start, self.y_end)

        plt.plot(self.x_values, self.y_values, 'o', label='Original data', markersize=4)
        plt.plot(self.x_smooth, self.y_smooth, '-', label='Smoothed trajectory', linewidth=2)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()

    # 对于没有读取到的数据, 速度设为0
    def calculate_average_speed(self, speeds):
        frame_number = max(len(self.all_centroids), len(speeds))

        average_speed = sum(speeds) / frame_number
        average_speed = round(average_speed, 2)

        return average_speed

    def run(self, visualize=True, analyze=False):
        self.update_all_centroids()  # 找每一帧的 centroid, 更新x,y值
        self.x_values, self.y_values = self.coordinates_to_xy(self.CENTROIDS)

        # smooth the track
        self.x_smooth = csaps(self.TIME, self.x_values, self.TIME, smooth=self.SMOOTH_RATE)
        self.y_smooth = csaps(self.TIME, self.y_values, self.TIME, smooth=self.SMOOTH_RATE)

        self.calculate_speed_variance()
        self.speed_variance_smoothed = self.calculate_speed_variance_smoothed(self.x_smooth, self.y_smooth, self.TIME)

        self.calculate_velocity_variance()
        self.velocity_variance_smoothed = self.calculate_velocity_variance_smoothed(self.x_smooth, self.y_smooth,
                                                                                    self.TIME)

        self.stillness_percentage = round((1 - (len(self.CENTROIDS) / len(self.all_centroids))), 3)
        stillness_percentage_print = f"{self.stillness_percentage:.1%}"

        self.impute()
        self.calculate_approaching_tendency()
        self.calculate_time_within_suspicious_range()

        suspicious_time_percent_print = f"{self.suspicious_time_percent:.1%}"

        anomaly_percent = (self.filtered_record_all / len(self.all_centroids))
        anomaly_percent = round(anomaly_percent, 2)
        self.generate_statistics()

        if analyze:
            print(f'SPEED: {self.SPEED}')
            print(f'SPEED after smooth: {self.SPEED_SMOOTH}')
            print(f"maximum speed: {max(self.SPEED)}")
            print(f"average speed: {round(average(self.SPEED), 2)}")
            print(f"the average speed after smooth: {round(average(self.SPEED_SMOOTH), 2)}")

            print("")

            print(f'the speed variance is: {round(self.speed_variance, 2)}')
            print(f'the speed_smoothed variance is: {round(self.speed_variance_smoothed, 2)}')

            print("")

            print(f"Velocity: {self.velocity}")
            print(f"Velocity after smooth: {self.velocity_smooth}")

            print("")

            print(f'the velocity variance is: {round(self.velocity_variance, 2)}')
            print(f'the velocity_smoothed variance is: {round(self.velocity_variance_smoothed, 2)}')

            print("")

            print(f"stillness percentage: {stillness_percentage_print}")

            print("")

            print(f"approaching_trend: {self.approaching_tendency}")
            print(f"percent of time within suspicious range: {suspicious_time_percent_print}")

            print(f"anomaly percent: {anomaly_percent}")

        if visualize:
            self.show_plot()

        statics = [self.average_speed,
                   self.speed_variance,
                   round(average(self.SPEED_SMOOTH), 2),
                   round(self.speed_variance_smoothed, 2),
                   self.velocity_variance,
                   round(self.velocity_variance_smoothed, 2),
                   self.stillness_percentage,
                   self.approaching_tendency,
                   round(self.suspicious_time_percent, 3),
                   anomaly_percent]

        print(statics)


"""
class SpeedMethods:
    @staticmethod
    def calculate_speed(coordinate1, coordinate2, time1, time2):
        x1, y1 = coordinate1[0], coordinate1[1]
        x2, y2 = coordinate2[0], coordinate2[1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        speed = distance / (time2 - time1)
        return round(speed, 2)

    @staticmethod
    def calculate_speed_smooth(x_values, y_values, timestamps):
        for i in range(len(x_values) - 1):
            x1, x2 = x_values[i], x_values[i + 1]
            y1, y2 = y_values[i], y_values[i + 1]
            time1, time2 = timestamps[i], timestamps[i + 1]

            # 计算速度
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            speed = distance / (time2 - time1)

            SPEED_SMOOTH.append(round(speed, 2))

        # 计算方差
        variance = statistics.variance(SPEED_SMOOTH)
        return variance

    @staticmethod
    def calculate_speed_variance():
        for i in range(len(CENTROIDS) - 1):
            speed = SpeedMethods.calculate_speed(CENTROIDS[i], CENTROIDS[i + 1], TIME[i], TIME[i + 1])
            SPEED.append(speed)
        variance = statistics.variance(SPEED)
        return variance


class VelocityMethods:
    # 返回(v_x, v_y) 带正负号速度
    @staticmethod
    def calculate_velocity(coordinate1, coordinate2, time1, time2) -> tuple[float, float]:
        x1, y1 = coordinate1[0], coordinate1[1]
        x2, y2 = coordinate2[0], coordinate2[1]
        distance_x = x2 - x1
        distance_y = y2 - y1
        time = time2 - time1

        speed_x = round(distance_x / time)
        speed_y = round(distance_y / time)
        velocity = speed_x, speed_y

        return velocity

    @staticmethod
    def calculate_velocity_variance():
        for i in range(len(CENTROIDS) - 1):
            speed = SpeedMethods.calculate_speed(CENTROIDS[i], CENTROIDS[i + 1], TIME[i], TIME[i + 1])
            SPEED.append(speed)
        variance = statistics.variance(SPEED)
        return variance

"""
if __name__ == '__main__':
    data = FRAMES
    processor = Preprocessing(data)
    processor.run()
