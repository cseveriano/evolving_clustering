'''
Based on code available at:
https://github.com/FelixNeutatz/CluStream
'''


import numpy as np
import sys
from sklearn.cluster import KMeans
import random


class CluStream:
    # q - maximum number of clusters
    #
    # m -      approximation detail for timestamp
    #          m last data points of the micro-cluster (equal to max clusters)
    #
    # delta -  When the least relevance stamp of any micro-cluster is below a user-defined threshold delta,
    #          it can be eliminated and a new micro-cluster can be created with a unique id
    #          corresponding to the newly arrived data point
    # radius_factor -       The maximum boundary of the micro-cluster is defined as
    #          a factor of t of the RMS deviation of the data points in the cluster from the centroid.
    #          We define this as the maximal boundary factor.

    def __init__(self, q=2, m=10, radius_factor = 1.8, delta=10, k=5, init_number=100):
        self.timestep = 0

        self.q = q

        self.radius_factor = radius_factor

        # heuristic_factor - heuristic maximal boundary factor
        #          default value = 2
        #          For a cluster which contains only 1 point,
        #          the maximum boundary is choosen to be r times the maximum boundary of the next closest cluster.
        self.heuristic_factor = 2
        self.m = m
        self.delta = delta
        self.init_number = init_number
        self.micro_clusters = [None] * self.q
        self.k = k
        self.macro_clusters = [None] * self.k
        self.micro_cluster_labels = []

    def get_nearest_micro_cluster(self, sample):
        smallest_distance = sys.float_info.max


        nearest_micro_cluster = None
        nearest_micro_cluster_index = -1
        for i, micro_cluster in enumerate(self.micro_clusters):
            mc_center = micro_cluster[1] / micro_cluster[4]
            current_distance = np.linalg.norm(mc_center - sample)

            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_cluster = micro_cluster
                nearest_micro_cluster_index = i
        return nearest_micro_cluster_index, nearest_micro_cluster

    def fit(self, data):
            self.update_micro_clusters(data)
            self.update_macro_clusters()

    def predict(self,data):
        y = []

        for sample in data:
            index, _ = self.get_nearest_micro_cluster(sample)
            y.append(self.micro_cluster_labels[index])

        return y

    def fit_predict(self,data):
        self.fit(data)
        return self.predict(data)

    def update_micro_clusters(self, data):
        if self.timestep == 0:
            starter_dataset = data[0:self.init_number, :]
            stream_data = data[self.init_number:len(data), :]

            self.init_micro_clusters(starter_dataset)

        for i in range(0, len(stream_data)):
            Xik = np.array(stream_data[i])
            # get minimum euclidean distance cluster
            #                              centroid M = CF1x / n
            dist = [np.linalg.norm(Xik - (cluster[1] / cluster[4])) for cluster in self.micro_clusters]

            dist_sorted = np.argsort(dist)

            cluster_id = dist_sorted[0]

            n = self.micro_clusters[cluster_id][4]

            if n > 1:
                # RMS deviation
                squared_sum = np.square(self.micro_clusters[cluster_id][1])
                sum_of_squared = self.micro_clusters[cluster_id][0]

                RMSD = np.sqrt(np.abs(sum_of_squared - (squared_sum / n)))

                maximal_boundary = np.linalg.norm(RMSD) * self.radius_factor

                if i > 0:
                    maximal_boundary *= self.heuristic_factor
            else:
                # SPECIAL CASE: boundary is the closest distance to other micro-cluster
                cluster_center = self.micro_clusters[cluster_id][1] / self.micro_clusters[cluster_id][4]
                alt_dist = [np.linalg.norm(cluster_center - (cluster[1] / cluster[4])) for cluster in self.micro_clusters]
                alt_dist_sorted = np.sort(alt_dist)
                maximal_boundary = alt_dist_sorted[1] # gets the second element since the first is himself (dist=0)

            if dist[cluster_id] <= maximal_boundary:  # data point falls within the maximum boundary of the micro-cluster
                # data point is added to the micro-cluster
                self.micro_clusters[cluster_id] = self.micro_clusters[cluster_id] + \
                                               np.array(
                                                   [np.square(Xik), Xik, np.square(self.timestep), self.timestep, 1])
            else:  # create a new micro-cluster
                # determine if it is safe to delete any of the current micro-clusters as outliers
                mean_timestamp = [(cluster[3] / cluster[4]) for cluster in self.micro_clusters]
                standard_deviation_timestamp = [
                    np.sqrt((cluster[2] / cluster[4]) - np.square((cluster[3] / cluster[4]))) \
                    for cluster in self.micro_clusters]

                Z = []
                for i in range(0, len(self.micro_clusters)):
                    mc = self.m
                    if mc > self.micro_clusters[i][4]:
                        mc = self.micro_clusters[i][4]

                    #                    Z.append(self.m / (2 * self.micro_clusters[i][4]))
                    Z.append(mc / (2 * self.micro_clusters[i][4]))

                Z = np.array(Z)

                relevance_stamp = mean_timestamp + Z * standard_deviation_timestamp

                least_recent_cluster = np.argmin(relevance_stamp)


                if relevance_stamp[
                    least_recent_cluster] < self.delta:  # eliminate old cluster and create a new micro-cluster
                    self.micro_clusters[least_recent_cluster] = np.array(
                        [np.square(Xik), Xik, np.square(self.timestep), self.timestep, 1])
                else:  # merge the two micro-clusters which are closest to one another
                    # search for two closest clusters
                    minA_id = -1
                    minB_id = -1
                    min_dist = float("inf")
                    for a in range(0, len(self.micro_clusters)):
                        for b in range(a + 1, len(self.micro_clusters)):
                            d = np.linalg.norm((self.micro_clusters[b][1] / self.micro_clusters[b][4]) - \
                                               (self.micro_clusters[a][1] / self.micro_clusters[a][4]))
                            if d < min_dist:
                                minA_id = a
                                minB_id = b
                                min_dist = d
                    # merge them
                    self.micro_clusters[minA_id] = self.micro_clusters[minA_id] + self.micro_clusters[minB_id]

                    # create new cluster
                    self.micro_clusters[minB_id] = np.array(
                        [np.square(Xik), Xik, np.square(self.timestep), self.timestep, 1])

    def init_micro_clusters(self, starter_dataset):
        kmeans = KMeans(init='k-means++', n_clusters=self.q).fit(starter_dataset)

        starter_clusters = kmeans.cluster_centers_


        for i in range(0, len(starter_dataset)):
            starter_point = np.array(starter_dataset[i])
            # get minimum euclidean distance cluster
            dist = [np.linalg.norm(starter_point - cluster) for cluster in starter_clusters]
            cluster_id = np.argmin(dist)

            # add to micro-cluster
            cluster_tuple = np.array([np.square(starter_point), starter_point, np.square(i), i, 1])
            if (self.micro_clusters[cluster_id] == None):
                self.micro_clusters[cluster_id] = cluster_tuple
            else:
                self.micro_clusters[cluster_id] = self.micro_clusters[cluster_id] + cluster_tuple

    def update_macro_clusters(self):
        micro_cluster_centroids = [cluster[1] for cluster in self.micro_clusters]
        kmeans = KMeans(init='k-means++', n_clusters=self.k, n_init=10)
        kmeans.fit(micro_cluster_centroids)

        self.macro_clusters = kmeans.cluster_centers_
        self.micro_cluster_labels = kmeans.labels_




