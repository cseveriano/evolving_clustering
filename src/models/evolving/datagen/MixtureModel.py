import numpy as np

class MixtureModel:

    def __init__(self, dimensions, num_models, mixture_model_pre = None, target_drift = None):
        self.dimensions = dimensions
        self.num_models = num_models
        self.weights = np.zeros(self.num_models)
        self.means = []
        self.covariances = []

        range = self.num_models

        for i in np.arange(self.num_models):
            self.weights[i] = np.random.rand()
            mean = np.zeros(self.dimensions)

            for j in np.arange(self.dimensions):
                mean[j] = (np.random.rand() * range) - (range / 2.0)

            self.means.append(mean)
            self.covariances.append(self.generate_covariance())

        self.normalize_weights()

        if mixture_model_pre is not None and target_drift is not None:
            self.apply_drift(mixture_model_pre, target_drift)

    def apply_drift(self, mixture_model_pre, target_drift):
        adjust_factor = target_drift ** 2.0
        mean = np.zeros(self.dimensions)
        for i in np.arange(self.num_models):
            if i < mixture_model_pre.get_num_models():
                self.weights[i] = (self.weights[i] * adjust_factor) + (
                        mixture_model_pre.get_weight[i] * (1 - adjust_factor))

            for j in np.arange(self.dimensions):
                mean[j] = (self.get_mean(i)[j] * adjust_factor) + (
                        mixture_model_pre.get_mean(i)[j] * (1 - adjust_factor))
        self.normalize_weights()

    def normalize_weights(self, weightSum):
        weightSum = sum(self.weights)
        self.weights = [w / weightSum for w in self.weights]

    def generate_covariance(self):
        d = self.dimensions
        x = np.zeros((d, d))
        covariances = np.zeros((d, d))
        matrixSum = 0.0

        #Generate a random number between -1 and 1
        for i in np.arange(d):
            for j in np.arange(d):
                x[i][j] = (((np.random.rand() * 2.0)-1.0)+((np.random.rand() * 2.0)-1.0)) / 2.0

        # TODO: verificar a dinamica desse codigo, baseado em MMG de Richard Hugh Moulton
        for j in np.arange(d):
            for k in np.arange(j+1):
                for l in np.arange(d):
                    matrixSum += x[j][l] * x[k][l]

                covariances[j][k] = matrixSum
                covariances[k][j] = matrixSum
                matrixSum = 0.0

        return covariances

    def get_weights(self):
        return self.weights

    def get_weight(self, index):
        return self.weights[index]

    def get_mean(self, index):
        return self.means[index]

    def set_mean(self, index, mean):
        self.means[index] = mean

    def get_covariance(self, index):
        return self.covariances[index]

    def get_num_models(self):
        return self.num_models
