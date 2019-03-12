import numpy as np

class MixtureModel:

    def __init__(self, dimensions, num_models):
        self.dimensions = dimensions
        self.num_models = num_models
        self.weights = np.zeros(self.num_models)
        self.models = []


        range = self.num_models

        weightSum = 0.0
        means = np.zeros(self.dimensions)

        for i in np.arange(self.num_models):
            self.weights[i] = np.random.rand()
            weightSum += self.weights[i]

            for j in np.arange(self.dimensions):
                means[j] = (np.random.rand() * range) - (range / 2.0)


            covariances = self.generate_covariance()
            self.models.append((means, covariances))

        for i in np.arange(self.num_models):
            self.weights[i] = self.weights[i] / weightSum


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
