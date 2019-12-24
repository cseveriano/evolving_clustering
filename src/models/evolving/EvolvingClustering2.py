import math
import numpy as np
import networkx as nx
from numba import jit
import copy

class Teda:
    def __init__(self):
        self.curr_observation = []
        self.curr_mean = []
        self.curr_scal = 0.0
        self.curr_var = 0.0
        self.curr_var = 0.0
        self.curr_eccentricity = 0.0
        self.curr_typicality = 0.0
        self.curr_norm_eccentricity = 0.0
        self.curr_norm_typicality = 0.0
        self.outlier = False
        self.ecc_threshold = 0.0
        self.next_k = 0


class MicroCluster:
    def __init__(self):
        self.cont = 0
        self.nclusters = 0
        self.microChanged = 0
        self.teda = None
        self.raios = 0.0
        self.centros = None
        self.dens = 0.0
        self.dens2 = 0.0
        self.tips = 0.0
        self.cont = 0

class MacroCluster:
    def _init_(self):
        self.nclust = 0
        self.macro_list = []
        self.adj_matrix = None
        self.out = 0
        self.macro2 = None



class EvolvingClustering2:
    def __init__(self, rad=0.04, debug=False):
        self.out = 0
        self.rad = rad
        self.micro_obj = None
        self.macro_obj = None
        self.nclusters = 0
        self.raios = None
        self.teda = []
        self.tips = None
        self.debug = debug

    def fit(self, X):

        lenx = len(X)
        i = 1
        if self.debug:
            print("Training...")

        for xk in X:
            if self.debug:
                print("Training ", i , " of ", lenx)
            self.microteda_update(xk)
            self.macro_cluster_update(xk)
            i += 1

    def teda_mixture(self, xk, micro_list):

        n = len(micro_list)

        tips = np.zeros(n)
        dens = np.zeros(n)

        aux_cont = 0
        for i in micro_list:
            m = self.sigmoid(self.micro_obj.cont[i])
            teda_obj = self.rec_tedaClus(xk, m,
                              self.micro_obj.teda[i].curr_mean,
                              self.micro_obj.teda[i].curr_scal,
                              self.micro_obj.teda[i].next_k)

            tips[aux_cont] = teda_obj.curr_typicality
            if teda_obj.curr_norm_eccentricity != 0:
                dens[aux_cont] = 1/teda_obj.curr_norm_eccentricity
            else:
                dens[aux_cont] = 0

            aux_cont += 1

        tips[np.isnan(tips)] = 0
        dens[np.isnan(dens)] = 0

        dens = (dens) / (sum((dens)) + 0.00001)

        tip = sum(tips * dens)

        return tip

    def predict(self, X):
        yhat = np.zeros(len(X))
        for ydx, xk in enumerate(X):
            nmacro = self.macro_obj.macro2.nclust
            nmicro = self.micro_obj.nclusters

            tips = np.zeros(nmacro)
            for i, micro_list in enumerate(self.macro_obj.macro2.macro_list):
                tips[i] = self.teda_mixture(xk, micro_list)

            yhat[ydx] = np.argmax(tips)
        return yhat

    def microteda_update(self, xk):
        cont = 0
        if self.micro_obj is None:

            #create first cluster

            m = self.sigmoid(1)
            self.micro_obj = MicroCluster()
            self.micro_obj.cont = np.array([1])
            self.micro_obj.nclusters = 1
            self.micro_obj.microChanged = [1]
            teda = self.rec_tedaClus(xk, m)

            self.micro_obj.teda = np.array([teda])
            self.micro_obj.raios = np.array([math.sqrt(self.micro_obj.teda[0].curr_var)])
            self.micro_obj.centros = np.array([self.micro_obj.teda[0].curr_mean])
            self.micro_obj.dens = np.ones(1)
            self.micro_obj.dens2 = np.ones(1)
            self.micro_obj.tips = np.array([self.micro_obj.teda[0].curr_typicality])

        else:
            # calculate teda of x_i for all existing micro clusters
            outlier_cont = 0

            # allocate the amount of micro clusters
            updated_micro = [0] * self.micro_obj.nclusters
            for i in np.arange(self.micro_obj.nclusters):

                m = self.sigmoid(self.micro_obj.teda[i].next_k)
                teda = self.rec_tedaClus(xk, m,
                                    self.micro_obj.teda[i].curr_mean,
                                    self.micro_obj.teda[i].curr_scal,
                                    self.micro_obj.teda[i].next_k)

                #if its an outlier...
                if teda.outlier:
                    outlier_cont += 1
                else:
                    cont += 1
                    self.micro_obj.teda[i] = teda
                    self.micro_obj.cont[i] += 1
                    self.micro_obj.raios[i] = math.sqrt(self.micro_obj.teda[i].curr_var)
                    self.micro_obj.centros[i, :] = self.micro_obj.teda[i].curr_mean
                    self.micro_obj.dens2[i] = self.micro_obj.cont[i]/(self.micro_obj.raios[i]+0.0001)
                    self.micro_obj.dens[i] = 1/(self.micro_obj.teda[i].curr_norm_eccentricity+0.0001)
                    updated_micro[cont-1] = i
                    self.micro_obj.tips[i] = teda.curr_typicality

            if outlier_cont == self.micro_obj.nclusters:
                # if x_i is outlier for all existing micro clusters then create a new one
                newidx = self.micro_obj.nclusters # because arrays start with zero index
                self.micro_obj.nclusters += 1
                self.micro_obj.cont = np.append(self.micro_obj.cont, 1)
                self.micro_obj.microChanged = [newidx]
                m = self.sigmoid(1)
                self.micro_obj.teda = np.append(self.micro_obj.teda,self.rec_tedaClus(xk, m=m))
                self.micro_obj.raios = np.append(self.micro_obj.raios, math.sqrt(self.micro_obj.teda[newidx].curr_var))
                self.micro_obj.centros = np.vstack((self.micro_obj.centros,np.array([self.micro_obj.teda[newidx].curr_mean])))  # alocar inicialmente
                self.micro_obj.dens = np.append(self.micro_obj.dens,0)
                self.micro_obj.dens2 = np.append(self.micro_obj.dens2,0)
                self.micro_obj.tips = np.append(self.micro_obj.tips, self.micro_obj.teda[newidx].curr_typicality)

        self.micro_obj.micro_idx = np.arange(self.micro_obj.nclusters)
        if cont > 0:
            self.micro_obj.microChanged = updated_micro[:cont]

    def sigmoid(self, val, lambd =0.007, scala=3):
        return scala * (1 / (1 + math.exp(-lambd * (val-100))))

    def rec_tedaClus(self, curr_observation, m, previous_mean = None,
                          previous_scal = 0, k = 1):

        if previous_mean is None:
            previous_mean = curr_observation

        rec_mean = (((k - 1) / k) * previous_mean) + ((1 / k) * curr_observation)

        c_obs = np.array(curr_observation)
        rec_scal = EvolvingClustering2.calculate_scal(c_obs, k, previous_scal)

        r_mean = np.array(rec_mean)
        rec_var = EvolvingClustering2.calculate_variance(r_mean, rec_scal)

        if k == 1:
            rec_ecc = 0
        elif rec_var == 0 and k > 1:
            rec_ecc = 0
        elif k < 3 and rec_var > 0.001:
            rec_ecc = 1000
        else:
            r_diff = np.array(rec_mean - curr_observation)
            rec_ecc = EvolvingClustering2.calculate_eccentricity(k, r_diff, rec_var)

        ret_obj = Teda()
        ret_obj.curr_observation = curr_observation
        ret_obj.curr_mean = rec_mean
        ret_obj.curr_scal = rec_scal

        if k == 1:
            ret_obj.curr_var = 0.0
        else:
            ret_obj.curr_var = rec_var

        ret_obj.curr_eccentricity = rec_ecc
        ret_obj.curr_typicality = 1 - rec_ecc
        ret_obj.curr_norm_eccentricity = ret_obj.curr_eccentricity / 2

        if k != 2:
            ret_obj.curr_norm_typicality = ret_obj.curr_typicality / (k - 2)
        else:
            ret_obj.curr_norm_typicality = 0

        ret_obj.outlier = bool(ret_obj.curr_norm_eccentricity > ((m ** 2 + 1) / (2 * k))) # 5/k = 3 sd
        ret_obj.ecc_threshold = 1 / k
        ret_obj.next_k = k + 1

        return ret_obj

    @staticmethod
    @jit(nopython=True)
    def calculate_scal(c_obs, k, previous_scal):
        rec_scal = ((k - 1) / k) * previous_scal + (np.dot(c_obs, c_obs)) / k
        return rec_scal

    @staticmethod
    @jit(nopython=True)
    def calculate_variance(r_mean, rec_scal):
        rec_var = rec_scal - (np.dot(r_mean, r_mean))
        return rec_var

    @staticmethod
    @jit(nopython=True)
    def calculate_eccentricity(k, r_diff, rec_var):
        rec_ecc = (1 / k) + (np.dot(r_diff, r_diff)) / (k * rec_var)
        return rec_ecc

    def macro_cluster_update(self, xk):
        if self.macro_obj is None:
            self.macro_obj = MacroCluster()
            self.macro_obj.nclust = 1
            self.macro_obj.macro_list = []
            self.macro_obj.macro_list.append(1)
            self.macro_obj.adj_matrix = np.array([1])
            self.macro_obj.out = 0
            self.macro_obj.macro2 = copy.deepcopy(self.macro_obj)
        else:
            idxs = self.micro_obj.microChanged
            centros = self.micro_obj.centros
            raios = self.micro_obj.raios
            n = len(idxs)
            m = self.micro_obj.nclusters
            d = len(centros[0])
            adj_matrix = self.macro_obj.adj_matrix

#            dists = np.zeros((n, m))

            # calculate distances of micro clusters that changed for every center
            if m == 1:
                dists = np.zeros(1)
                adj = np.ones((1,1))
            else:
                dists = [[EvolvingClustering2.calculate_distance(changed_center, center) for center in centros] for changed_center in centros[idxs, :]]

                # adjacency matrix
                raioschanged = np.tile(np.array(raios[idxs]).reshape((n, 1)), m)
                all_raios = np.tile(np.array(raios), (n,1))
                adj = 1 * (dists <= (2 * (raioschanged + all_raios)))

            if m != len(adj_matrix):
                # insere uma coluna e uma linha de zeros
                adj_matrix = np.c_[adj_matrix, np.zeros((adj_matrix.shape[0],1))]
                adj_matrix = np.vstack((adj_matrix, np.zeros((1,len(adj_matrix[0])))))


            if sum([i in idxs for i in self.micro_obj.micro_idx]) == m:
                    adj_matrix = adj
            else:
                adj_matrix[idxs,:] = adj
                adj_matrix[:, idxs] = adj.T

            grafo2 = nx.from_numpy_matrix(adj_matrix)
            self.macro_obj.macro_list = list(nx.connected_components(grafo2))
            self.macro_obj.nclust = len(self.macro_obj.macro_list)

            # check for outliers
            outs = np.array([False] * m, dtype=bool)
            for i in np.arange(self.macro_obj.nclust):
                auxlist = list(self.macro_obj.macro_list[i])
                mdens = np.mean(self.micro_obj.dens[auxlist])
                outs[auxlist] = [bool((d < mdens) or (c <= 2)) for d, c in zip(self.micro_obj.dens[auxlist], self.micro_obj.cont[auxlist])]

            micro2 = MicroCluster()
            micro2.nclusters = sum(outs == False)
            micro2.dens = self.micro_obj.dens[outs == False]
            micro2.tips = self.micro_obj.tips[outs == False]
            micro2.micro_idx = self.micro_obj.micro_idx[outs == False]

            macro2_obj = MacroCluster()

            macro2_obj.nclust = 0
            macro2_obj.macro_list = []
            macro2_obj.typicallity = None
#            centros2 = centros[outs == 0,:]
#            raios2 = raios[outs == 0]

            adj_matrix2 = adj_matrix[outs == False, :][:, outs == False]
            if (adj_matrix2 is not None and adj_matrix2.size != 0):
                if len(adj_matrix2.shape) == 1:
                    adj_matrix2 = np.reshape(adj_matrix2,(adj_matrix2.shape[0], 1))
                grafo2 = nx.from_numpy_matrix(adj_matrix2)
                macro2_list = list(nx.connected_components(grafo2))

                if macro2_list is not None:
                    macro2_obj.nclust = len(macro2_list)
                    macro2_obj.typicallity = [0] *  macro2_obj.nclust

                    for i,mcs in enumerate(macro2_list):
                        macro2_list[i] = micro2.micro_idx[list(mcs)]
                        dens = micro2.dens[list(mcs)] / sum(micro2.dens[list(mcs)])
                        tips = micro2.tips[list(mcs)]
                        macro2_obj.typicallity[i] = sum(tips * dens)

                    macro2_obj.typicallity = macro2_obj.typicallity / sum(macro2_obj.typicallity)
                    macro2_obj.macro_list = macro2_list

                else:
                    macro2_obj.nclust = 0


            self.macro_obj.out = outs
            self.macro_obj.adj_matrix = adj_matrix
            self.macro_obj.macro2 = macro2_obj
            self.macro_obj.micro2 = micro2

        pass

    @staticmethod
    @jit(nopython=True)
    def calculate_distance(changed_center, center):
        return np.linalg.norm(changed_center - center)