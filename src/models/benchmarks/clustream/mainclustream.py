import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from scipy import stats

#############

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

#print data
data =  np.asmatrix(data)

np.random.shuffle(data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))




kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10).fit(data)
kmeans.fit(data)

print(len(kmeans.cluster_centers_))

#############

# The value of InitNumber is chosen to be as large as permitted by the computational complexity of a k-means
# algorithm creating q clusters.
InitNumber = 100

q = 20 #number of micro-clusters

starter_dataset = data[0:InitNumber,:]
stream_data = data[InitNumber:len(data),:]

kmeans = KMeans(init='k-means++', n_clusters=q).fit(starter_dataset)

starter_clusters = kmeans.cluster_centers_

micro_clusters = [None] * q

for i in range(0, len(starter_dataset)):
    starter_point = np.array(starter_dataset[i])
    # get minimum euclidean distance cluster
    dist = [np.linalg.norm(starter_point - cluster) for cluster in starter_clusters]
    cluster_id = np.argmin(dist)

    # add to micro-cluster
    cluster_tuple = np.array([np.square(starter_point), starter_point, np.square(i), i, 1])
    if (micro_clusters[cluster_id] == None):
        micro_clusters[cluster_id] = cluster_tuple
    else:
        micro_clusters[cluster_id] = micro_clusters[cluster_id] + cluster_tuple


def calcSSQ(points, centroids, horizon):
    SSQ = 0
    for point in points[(-1 * horizon):]:
        dist = [np.linalg.norm(point - cluster) for cluster in starter_clusters]
        SSQ += np.square(min(dist))
    print
    "SSQ: " + str(SSQ)

    return SSQ


#          The maximum boundary of the micro-cluster is defined as
#          a factor of t of the RMS deviation of the data points in the cluster from the centroid.
#          We define this as the maximal boundary factor.
#
# r -      heuristic maximal boundary factor
#          default value = 2
#          For a cluster which contains only 1 point,
#          the maximum boundary is choosen to be r times the maximum boundary of the next closest cluster.
#
# m -      approximation detail for timestamp
#          m last data points of the micro-cluster
#
# delta -  When the least relevance stamp of any micro-cluster is below a user-defined threshold delta,
#          it can be eliminated and a new micro-cluster can be created with a unique id
#          corresponding to the newly arrived data point

def CluStream(record, timestep, micro_clusters, t=2, r=2, m=10, delta=10):
    Xik = np.array(record)
    # get minimum euclidean distance cluster
    #                              centroid M = CF1x / n
    dist = [np.linalg.norm(Xik - (cluster[1] / cluster[4])) for cluster in micro_clusters]

    dist_sorted = np.argsort(dist)

    cluster = dist_sorted[0]

    i = 0
    while True:
        cluster_id = dist_sorted[i]

        n = micro_clusters[cluster_id][4]

        if n > 1:
            # RMS deviation
            squared_sum = np.square(micro_clusters[cluster_id][1])
            sum_of_squared = micro_clusters[cluster_id][0]

            RMSD = np.sqrt(np.abs(sum_of_squared - (squared_sum / n)))

            maximal_boundary = np.linalg.norm(RMSD) * t

            if i > 0:
                maximal_boundary *= r

            break

        # find next closest cluster
        i += 1

    if dist[cluster] <= maximal_boundary:  # data point falls within the maximum boundary of the micro-cluster
        # data point is added to the micro-cluster
        micro_clusters[cluster] = micro_clusters[cluster] + \
                                  np.array([np.square(Xik), Xik, np.square(timestep), timestep, 1])
        print
        "add to cluster"
    else:  # create a new micro-cluster
        # determine if it is safe to delete any of the current micro-clusters as outliers
        mean_timestamp = [(cluster[3] / cluster[4]) for cluster in micro_clusters]
        standard_deviation_timestamp = [np.sqrt((cluster[2] / cluster[4]) - np.square((cluster[3] / cluster[4]))) \
                                        for cluster in micro_clusters]

        Z = []
        for i in range(0, len(micro_clusters)):
            mc = m
            if mc > micro_clusters[i][4]:
                mc = micro_clusters[i][4]

            Z.append(m / (2 * micro_clusters[i][4]))

        Z = np.array(Z)

        relevance_stamp = mean_timestamp + Z * standard_deviation_timestamp

        least_recent_cluster = np.argmin(relevance_stamp)

        print
        relevance_stamp[least_recent_cluster]

        if relevance_stamp[least_recent_cluster] < delta:  # eliminate old cluster and create a new micro-cluster
            micro_clusters[least_recent_cluster] = np.array([np.square(Xik), Xik, np.square(timestep), timestep, 1])
            print
            "eliminated cluster"
        else:  # merge the two micro-clusters which are closest to one another
            # search for two closest clusters
            minA_id = -1
            minB_id = -1
            min_dist = float("inf")
            for a in range(0, len(micro_clusters)):
                for b in range(a + 1, len(micro_clusters)):
                    d = np.linalg.norm((micro_clusters[b][1] / micro_clusters[b][4]) - \
                                       (micro_clusters[a][1] / micro_clusters[a][4]))
                    if d < min_dist:
                        minA_id = a
                        minB_id = b
                        min_dist = d
            # merge them
            micro_clusters[minA_id] = micro_clusters[minA_id] + micro_clusters[minB_id]

            # create new cluster
            micro_clusters[minB_id] = np.array([np.square(Xik), Xik, np.square(timestep), timestep, 1])
            print
            "merged cluster"

    return micro_clusters



SSQ = []
for i in range(0, len(stream_data)):
    micro_clusters = CluStream(stream_data[i], i, micro_clusters)
    SSQ.append(calcSSQ((stream_data[0:(i+1)]), micro_clusters,100))