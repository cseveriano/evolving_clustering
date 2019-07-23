import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
from numba import jit

class EvolvingClustering:
    def __init__(self, macro_cluster_update=1,
                 verbose=0, variance_limit=0.001, debug=False, plot_graph=False):
        self.verbose = verbose
        self.total_num_samples = 0
        self.micro_clusters = []
        self.macro_clusters = []
        self.active_macro_clusters = []
        self.graph = nx.Graph()
        self.active_graph = nx.Graph()
        self.variance_limit = variance_limit
        self.macro_cluster_update = macro_cluster_update
        self.debug = debug
        self.plot_graph = plot_graph

    @staticmethod
    def get_micro_cluster(id, num_samples, mean, variance, density):
        return {"id": id,"num_samples": num_samples, "mean": mean, "variance": variance, "density": density, "active": True, "changed": True}

    def create_new_micro_cluster(self, x):

        id = len(self.micro_clusters)
        num_samples = 1
        mean = x
        variance = 0
        density = 0

        self.micro_clusters.append(EvolvingClustering.get_micro_cluster(id, num_samples, mean, variance, density))
        self.graph.add_node(id)

    def is_outlier(self, s_ik, var_ik, norm_ecc):

        if s_ik < 3:
            outlier = (var_ik > self.variance_limit)
        else:
            mik_sik = 3 / (1 + math.exp(-0.007 * (s_ik - 100)))
            outlier_limit = ((mik_sik ** 2) + 1) / (2 * s_ik)
            outlier = (norm_ecc > outlier_limit)

        return outlier

    @staticmethod
    def update_micro_cluster(micro_cluster, x, num_samples, mean, variance, norm_ecc):
        micro_cluster["num_samples"] = num_samples
        micro_cluster["mean"] = mean
        micro_cluster["variance"] = variance
        micro_cluster["density"] = 1 / norm_ecc
        micro_cluster["changed"] = True

    @staticmethod
    def get_updated_micro_cluster_values(x, s_ik, mu_ik, var_ik):

        s_ik += 1
        mean = ((s_ik - 1) / s_ik) * mu_ik + (x / s_ik)

        # Codigo dissertacao
        delta = x - mean
        variance = EvolvingClustering.update_variance(delta, s_ik, var_ik)

        norm_ecc = EvolvingClustering.get_normalized_eccentricity(x, s_ik, mean, variance)
        return (s_ik, mean, variance, norm_ecc)

    @staticmethod
    @jit(nopython=True)
    def update_variance(delta, s_ik, var_ik):
        variance = ((s_ik - 1) / s_ik) * var_ik + (np.linalg.norm(delta) ** 2 / (s_ik - 1))
        return variance

    @staticmethod
    def get_normalized_eccentricity(x, num_samples, mean, var):
        ecc = EvolvingClustering.get_eccentricity(x, num_samples, mean, var)
        return ecc / 2

    @staticmethod
    @jit(nopython=True)
    def get_eccentricity(x, num_samples, mean, var):
        if var == 0 and num_samples > 1:
            result = (1/num_samples)
        else:
            a = mean - x
            result = ((1 / num_samples) + (np.linalg.norm(a) ** 2 / (num_samples * var)))

        return result

    def fit(self, X, update_macro_clusters=True):

        lenx = len(X)

        if self.debug:
            print("Training...")

        for xk in X:
            self.update_micro_clusters(xk)

            self.total_num_samples += 1

            if self.debug:
                print('Training %d of %d' %(self.total_num_samples, lenx))

        if update_macro_clusters:
            if self.debug:
                print('Updating Macro_clusters')
            self.update_macro_clusters()

        if self.plot_graph:
            self.plot_micro_clusters(X)


    def update_micro_clusters(self, xk):
        # First sample
        if self.total_num_samples == 0:
            self.create_new_micro_cluster(xk)
        else:
            new_micro_cluster = True

            for mi in self.micro_clusters:
                mi["changed"] = False
                s_ik = mi["num_samples"]
                mu_ik = mi["mean"]
                var_ik = mi["variance"]

                (num_samples, mean, variance, norm_ecc) = EvolvingClustering.get_updated_micro_cluster_values(xk, s_ik, mu_ik, var_ik)

                if not self.is_outlier(num_samples, variance, norm_ecc):
                    self.update_micro_cluster(mi, xk, num_samples, mean, variance, norm_ecc)
                    new_micro_cluster = False

            if new_micro_cluster:
                self.create_new_micro_cluster(xk)

    def update_macro_clusters(self):
        self.define_macro_clusters()
        self.define_activations()

    def predict(self, X):
        self.labels_ = np.zeros(len(X), dtype=int)
        index = 0
        lenx = len(X)

        if self.debug:
            print('Predicting...')

        for xk in X:
            memberships = []
            for mg in self.active_macro_clusters:
                active_micro_clusters = self.get_active_micro_clusters(mg)

                memberships.append(EvolvingClustering.calculate_membership(xk, active_micro_clusters))

            self.labels_[index] = np.argmax(memberships)
            index += 1

            if self.debug:
                print('Predicting %d of %d' % (index, lenx))

        return self.labels_

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


    @staticmethod
    def calculate_micro_membership(x, params):

        micro_cluster = params[0]
        total_density = params[1]

        d = micro_cluster["density"]

        t = 1 - EvolvingClustering.get_normalized_eccentricity(x, micro_cluster["num_samples"], micro_cluster["mean"], micro_cluster["variance"])
        return (d / total_density) * t

    def get_total_density(self):
        active_mcs = self.get_all_active_micro_clusters()
        total_density = 0

        for m in active_mcs:
            total_density += m["density"]

        return total_density

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
        micro_clusters_pairs = list(it.combinations(self.micro_clusters, 2))

        # Create macro-clusters from intersected micro-clusters
        for m in micro_clusters_pairs:
            mi = m[0]
            mj = m[1]
            if mi["id"] != mj["id"] :
                edge = (mi["id"],mj["id"])
                if EvolvingClustering.has_intersection(mi, mj):
                    self.graph.add_edge(*edge)
                elif EvolvingClustering.nodes_connected(mi["id"],mj["id"], self.graph):
                    self.graph.remove_edge(*edge)

        self.macro_clusters = list(nx.connected_components(self.graph))




    @staticmethod
    def nodes_connected(u, v, G):
        return u in G.neighbors(v)

    def define_activations(self):

        self.active_graph = self.graph.copy()

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
                    self.active_graph.remove_node(mi["id"])

        self.active_macro_clusters = list(nx.connected_components(self.active_graph))


    @staticmethod
    def has_intersection(mi, mj):
        mu_i = mi["mean"]
        mu_j = mj["mean"]
        var_i = mi["variance"]
        var_j = mj["variance"]

        d = EvolvingClustering.get_euclidean_distance(mu_i, mu_j)
        dist = np.sqrt(np.sum(d))

        deviation = EvolvingClustering.get_deviation(var_i, var_j)

        return dist <= deviation

    @staticmethod
    @jit(nopython=True)
    def get_deviation(var_i, var_j):
        deviation = 2 * (np.sqrt(var_i) + np.sqrt(var_j))
        return deviation

    @staticmethod
    @jit(nopython=True)
    def get_euclidean_distance(mu_i, mu_j):
        dist = [(a - b) ** 2 for a, b in zip(mu_i, mu_j)]
        return dist

    def plot_micro_clusters(self, X):

        micro_clusters = self.get_all_active_micro_clusters()
        ax = plt.gca()

        ax.scatter(X[:, 0], X[:, 1], s=1, color='b')

        for m in micro_clusters:
            mean = m["mean"]
            std = math.sqrt(m["variance"])

            circle = plt.Circle(mean, std, color='r', fill=False)

            ax.add_artist(circle)
        plt.draw()