import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance

class EvolvingClustering:
    def __init__(self, macro_cluster_update=1000,
                 verbose=0, variance_limit=0.001, debug=False):
        self.verbose = verbose
        self.total_num_samples = 0
        self.micro_clusters = []
        self.macro_clusters = []
        self.graph = nx.Graph()
        self.variance_limit = variance_limit
        self.macro_cluster_update = macro_cluster_update
        self.debug = debug

    @staticmethod
    def get_micro_cluster(id, num_samples, mean, scal, variance, density):
        return {"id": id,"num_samples": num_samples, "mean": mean, "scal": scal, "variance": variance, "density": density, "active": True, "changed": True}

    def create_new_micro_cluster(self, x):

        id = len(self.micro_clusters)
        num_samples = 1
        mean = x
        scal = np.dot(x,x)
        variance = 0
        density = 0

        self.micro_clusters.append(EvolvingClustering.get_micro_cluster(id, num_samples, mean, scal, variance, density))
        self.graph.add_node(id)

    def is_outlier(self, x, s_ik, mu_ik, var_ik):

        if s_ik == 2:
            outlier = (var_ik > self.variance_limit)
        else:
            mik_sik = 3 / (1 + math.exp(-0.007 * (s_ik - 100)))
            outlier_limit = ((mik_sik ** 2) + 1) / (2 * s_ik)
            norm_ecc = EvolvingClustering.get_normalized_eccentricity(x, s_ik, mu_ik, var_ik)
            outlier = (norm_ecc > outlier_limit)

        return outlier

    @staticmethod
    def update_micro_cluster(micro_cluster, x, num_samples, mean, scal, variance):
        micro_cluster["num_samples"] = num_samples
        micro_cluster["mean"] = mean
        micro_cluster["scal"] = scal
        micro_cluster["variance"] = variance
        norm_ecc = EvolvingClustering.get_normalized_eccentricity(x, num_samples, mean, variance)
        micro_cluster["density"] = 1 / norm_ecc
        micro_cluster["changed"] = True

    @staticmethod
    def get_updated_micro_cluster_values(x, micro_cluster):
        s_ik = micro_cluster["num_samples"]
        mu_ik = micro_cluster["mean"]
        scal_ik = micro_cluster["scal"]
        var_ik = micro_cluster["variance"]

        s_ik += 1
        mean = ((s_ik - 1) / s_ik) * mu_ik + (x / s_ik)
        scal = ((s_ik - 1) / s_ik) * scal_ik + (np.dot(x,x) / s_ik)

        # variance = scal - np.dot(mean, mean)
        delta = x - mean
        variance = ((s_ik - 1) / s_ik) * var_ik + (np.dot(delta, delta) / (s_ik - 1))
        return (s_ik, mean, scal, variance)

    @staticmethod
    def get_normalized_eccentricity(x, num_samples, mean, var):
        return EvolvingClustering.get_eccentricity(x, num_samples, mean, var) / 2

    @staticmethod
    def get_eccentricity(x, num_samples, mean, var):
        a = mean - x
        result = ((1/num_samples) + (np.dot(a, a) / (num_samples * (var))))
        return result

    def fit(self, X):

        inc_X = []
        lenx = len(X)

        if self.debug:
            print("Fitting...")

        for xk in X:
            inc_X.append(list(xk))
            self.update_micro_clusters(xk)
            if (self.total_num_samples > 0) and (self.total_num_samples % self.macro_cluster_update == 0):
                self.update_macro_clusters()

            self.total_num_samples += 1

            if self.debug:
                print('Fitting %d of %d' %(self.total_num_samples, lenx))

        self.plot_micro_clusters(X)

        self.predict_labels(X)

    def update_micro_clusters(self, xk):
        # First sample
        if self.total_num_samples == 0:
            self.create_new_micro_cluster(xk)
        else:
            new_micro_cluster = True

            for mi in self.micro_clusters:
                mi["changed"] = False
                (num_samples, mean, scal, variance) = EvolvingClustering.get_updated_micro_cluster_values(xk, mi)

                if not self.is_outlier(xk, num_samples, mean, variance):
                    self.update_micro_cluster(mi, xk, num_samples, mean, scal, variance)
                    new_micro_cluster = False

            if new_micro_cluster:
                self.create_new_micro_cluster(xk)

    def update_macro_clusters(self):
        self.define_macro_clusters()
        self.define_activations()

    def predict_labels(self, X):
        self.labels_ = np.zeros(len(X), dtype=int)
        index = 0
        lenx = len(X)

        if self.debug:
            print('Predicting...')

        for xk in X:
            memberships = []
            for mg in self.macro_clusters:
                active_micro_clusters = self.get_active_micro_clusters(mg)

                memberships.append(EvolvingClustering.calculate_membership(xk, active_micro_clusters))

            self.labels_[index] = np.argmax(memberships)
            index += 1

            if self.debug:
                print('Predicting %d of %d' % (index, lenx))

    @staticmethod
    def calculate_membership(x, active_micro_clusters):
        total_density = 0
        for m in active_micro_clusters:
            total_density += m["density"]

        mb = 0
        for m in active_micro_clusters:
            d = m["density"]

            t = 1 - EvolvingClustering.get_normalized_eccentricity(x, m["num_samples"], m["mean"], m["variance"])
            mb += (d / total_density) * t
        return mb

    def get_active_micro_clusters(self, mg):
        active_micro_clusters = []
        for mi_ind in mg:
            mi = self.micro_clusters[mi_ind]
            if mi["active"]:
                active_micro_clusters.append(mi)
        return active_micro_clusters

    def get_all_active_micro_clusters(self):
        active_micro_clusters = []

        for m in self.micro_clusters:
            if m["active"]:
                active_micro_clusters.append(m)
        return active_micro_clusters

    def get_changed_micro_clusters(self):
        changed_micro_clusters = []

        for m in self.micro_clusters:
            if m["changed"]:
                changed_micro_clusters.append(m)
        return changed_micro_clusters

    def define_macro_clusters(self):
        changed_micro_clusters = self.get_changed_micro_clusters()

        # Create macro-clusters from intersected micro-clusters
        for mi in changed_micro_clusters:
            for mj in self.micro_clusters:
                if EvolvingClustering.has_intersection(mi, mj):
                    self.graph.add_edge(mi["id"],mj["id"])

        self.macro_clusters = list(nx.connected_components(self.graph))

    def define_activations(self):
        for mg in self.macro_clusters:
            num_micro = len(mg)
            total_density = 0

            for i in mg:
                total_density += self.micro_clusters[i]["density"]

            mean_density = total_density / num_micro

            for i in mg:
                mi = self.micro_clusters[i]
                mi["active"] = (mi["num_samples"] > 2) and (mi["density"] >= mean_density)

                if not mi["active"]:
                    self.graph.remove_edges_from(list(self.graph.edges(mi["id"])))

    def partial_fit(self, X, y=None):
        return self

    def predict(self, X):
        labels = []

        return labels

    @staticmethod
    def has_intersection(mi, mj):
        mu_i = mi["mean"]
        mu_j = mj["mean"]
        var_i = mi["variance"]
        var_j = mj["variance"]

        dist = distance.euclidean(mu_i, mu_j)
        deviation = 2 * (math.sqrt(var_i) + math.sqrt(var_j))

        return dist <= deviation

    def plot_micro_clusters(self, X):

        micro_clusters = self.get_all_active_micro_clusters()
        ax = plt.gca()

        ax.scatter(X[:, 0], X[:, 1], s=10, color='b')

        for m in micro_clusters:
            mean = m["mean"]
            std = math.sqrt(m["variance"])

            circle = plt.Circle(mean, std, color='r', fill=False)

            ax.add_artist(circle)
        plt.draw()