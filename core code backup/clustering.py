from numpy import *
from data2 import *
import matplotlib.pyplot as plt


class KmeansClustering:

    def __init__(self, k, x_values, y_values):
        self.k = k
        self.x_values = x_values  # [x1, x2, x3, x4]
        self.y_values = y_values  # [y1, y2, y3, y4]

    # remove dots with impossible distance
    def _ghost_dots_removing(self, cluster):
        self.y_values
        pass


    # 两个centroids 列表 是否一样
    def _kmeans_centroids_converge(self, centroids1:list[tuple, tuple,], centroids2:list[tuple, tuple]) -> bool:
        if len(centroids1) != len(centroids2):
            return False
        for _ in centroids1:
            if _ not in centroids2:
                return False
        return True

    # 得到cluster的 中心点(平均值)
    @staticmethod
    def _kmeans_cluster_center(cluster: list[tuple, tuple, ...]) -> tuple[float, float]:
        x, y = 0, 0
        for coordinate in cluster:
            x += coordinate[0]
            y += coordinate[1]
        x = round(x/len(cluster), 3)
        y = round(y/len(cluster), 3)

        return x, y

    # 计算两点距离(参数是两个独立坐标)
    def _kmeans_distance(self, cor1:tuple, cor2:tuple) -> float:
        distance = ((cor1[0] - cor2[0]) ** 2 + (cor1[1] - cor2[1]) ** 2) ** 0.5
        return distance

    # 把所有x-y独立集合, 变成所有点的 坐标集合; 参数可以接受 numpy array
    def _kmeans_get_coordination(self, x_values, y_values) -> list[tuple, ...]:
        coordinates = []
        if len(x_values) == len(y_values):
            for i in range(len(x_values)):
                coordinates.append((x_values[i], y_values[i]))
            return coordinates

    # 计算出新的 centroid
    def _kmeans_updating_centroids(self, clusters, centroids, coordinates):
        iterate_round = 1  # iterate time, just for debug
        while True:
            # ①计算 每个点 到 每个centroid 的距离 ② 把点加到centroid 对应的 clustering1里
            for point in coordinates:
                distances = [self._kmeans_distance(point, centroids[i]) for i in range(self.k)]
                belonged_centroid_index = distances.index(min(distances))
                clusters[belonged_centroid_index].append(point)

            # 计算每个cluster的中心
            new_centroids = []
            for cluster in clusters:
                new_centroids.append(self._kmeans_cluster_center(cluster))

            # 如果不收敛,继续算
            if self._kmeans_centroids_converge(new_centroids, centroids):
                centroids = new_centroids
                # print(clusters[0])
                return centroids
            else:
                centroids = new_centroids
                iterate_round += 1

                for cluster in clusters:
                    cluster.clear()
                # print("当前循环次数: ", iterate_round, ", 目前centroid: ", centroids)


    # 初始化这一帧的clusters 和 centroids
    def _kmeans_clusters_initialize(self, coordinates: list) -> tuple[list[list, list, ...], list[tuple, tuple, ...]]:
        # clusters and centroids
        clusters = []  # [[cluster1], [cluster2], ```]
        centroids = []  # [centroid1, centroid2, ```]

        # initialize clusters
        for _ in range(self.k):
            centroid = coordinates.pop()
            centroids.append(centroid)
            clusters.append([])
            clusters[_].append(centroid)
        return clusters, centroids

    # 生成k个center,返回每个center的坐标
    # k 默认值问题
    # 如果所有目标都丢失了该怎么办? k>点数
    def kmeans_clustering(self) -> list[tuple, tuple, ...]:
        # 所有点坐标
        coordinates = self._kmeans_get_coordination(self.x_values, self.y_values)
        coordinates_for_centroid = coordinates.copy()
        point_quantity = len(coordinates)
        # 得到首次的 cluster & centroid
        clusters, centroids = self._kmeans_clusters_initialize(coordinates_for_centroid)

        final_centroids = self._kmeans_updating_centroids(clusters, centroids, coordinates)
        return final_centroids


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cluster = None

    def get_cluster(self):
        return self.cluster

    def set_cluster(self, index:int):
        self.cluster = index

    def get_coordinates(self):
        return self.x, self.y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y


class DensityBasedClustering:
    def __init__(self, x_values, y_values, min_points, eps):
        self.min_points = min_points
        self.eps = eps

        x_values, y_values = self._noise_remove(x_values, y_values)
        self.all_points = self._get_coordination(x_values, y_values)
        self.cores, self.borders, self.noises = self._points_classifying()

    # 去掉很近的不正确的点
    def _noise_remove(self, x_vals, y_vals):
        noise_indexes = []

        x_values = x_vals.tolist()
        y_values = y_vals.tolist()

        for y in y_values:
            if y < 0.1:
                noise_indexes.append(y_values.index(y))

        for index in noise_indexes[::-1]:
            x_values.pop(index)
            y_values.pop(index)

        return x_values, y_values

    def _distance(self, cor1:tuple, cor2:tuple) -> float:
        distance = ((cor1[0] - cor2[0]) ** 2 + (cor1[1] - cor2[1]) ** 2) ** 0.5
        return distance

    # 纯x,纯y 转坐标
    def _get_coordination(self, x_values, y_values) -> list[tuple, ...]:
        coordinates = []
        if len(x_values) == len(y_values):
            for i in range(len(x_values)):
                coordinates.append((x_values[i], y_values[i]))
            return coordinates

    # 给定范围内有多少个 points
    def _get_points_number_in_area(self, core:tuple, eps:float, coordinates:list) -> int:
        count = 0
        for point in coordinates:
            if self._distance(core, point) <= eps:
                count += 1
        # print(count)
        return count

    def _is_border_point(self, core_points:list, point:tuple) -> bool:
        for core in core_points:
            if self._distance(core, point) <= self.eps:
                return True
        return False

    # 把所有点分类
    def _points_classifying(self):
        unclassified = self.all_points.copy()
        cores = []
        borders = []
        noises = []

        # core points
        for point in unclassified:
            # print(f'{point}: ', end="")
            if self._get_points_number_in_area(point, self.eps, self.all_points) >= self.min_points:
                cores.append(point)
        for core in cores:
            unclassified.remove(core)

        # border points
        for point in unclassified:
            if self._is_border_point(core_points=cores, point=point):
                borders.append(point)
        for border in borders:
            unclassified.remove(border)

        noises = unclassified.copy()
        return cores, borders, noises

    def show_points(self):
        print(f'core points: {self.cores} \nborder points: {self.borders} \nnoises: {self.noises}')

    # 根据 core, border 创建cluster
    def clustering(self, merge=True):
        index = 0
        core_points = []
        border_points = []

        for point in self.cores:
            core_points.append(Point(point[0], point[1]))
        for point in self.borders:
            border_points.append(Point(point[0], point[1]))
        all_points = core_points + border_points

        if len(core_points) == 0:
            return None

        for p in core_points:
            if not p.get_cluster():
                index += 1
                p.set_cluster(index)
            for p2 in all_points:
                if not p2.get_cluster() and self._distance(p.get_coordinates(), p2.get_coordinates()) <= self.eps:
                    p2.set_cluster(index)

        clusters = []
        for _ in range(index):
            clusters.append([])

        for p in all_points:
            cluster_index = p.get_cluster()
            clusters[cluster_index-1].append(p.get_coordinates())

        if merge:
            result = []
            for _ in clusters:
                result += _
            clusters.clear()
            clusters.append(result)

        return clusters

    @staticmethod
    def get_center(cluster: list[tuple, tuple, ...]) -> tuple[float, float]:
        x, y = 0, 0
        for coordinate in cluster:
            x += coordinate[0]
            y += coordinate[1]
        x = round(x/len(cluster), 3)
        y = round(y/len(cluster), 3)

        return x, y

    def get_x_y_list(self, coordinates):
        x_val = []
        y_val = []
        for point in coordinates:
            x_val.append(point[0])
            y_val.append(point[1])
        return x_val, y_val


if __name__ == '__main__':
    """
    # test Kmeans
    x = [2.001, 1.999, 2.000, 2.005, 1.995, 2.010, 1.990, 2.002, 2.003, 1.995] # 2
    y = [1.501, 1.499, 1.505, 1.495, 1.500, 1.510, 1.490, 1.502, 1.503, 1.495] # 1.5

    k = KmeansClustering(1, x, y)
    point = k.kmeans_clustering()
    print(point)
    """


    # test density_based
    x_vals = [0, 100, 0, 100, 50, 49, 50, 49, 52]
    y_vals = [0, 100, 100, 0, 50, 49, 49, 50, 52]

    d = DensityBasedClustering(x_vals, y_vals, min_points=4, eps=3.1)
    print(d.clustering())

    # eps = 0.3, min = 3