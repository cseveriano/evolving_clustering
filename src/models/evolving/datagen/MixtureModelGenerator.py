import numpy as np

class MixtureModelGenerator:

    def __init__(self, num_features, num_classes):

        self.dimensions = num_features
        self.num_models = num_classes
        self.weights = np.zeros(self.num_models)
        self.models = []

    def prepare_for_use(self):

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


    def get_weighted_index(self):

        probSum = np.sum(self.weights)
        threshold = np.random.rand() * probSum
        index = 0
        partial = 0.0

        while partial <= threshold and index < len(self.weights):
            partial += self.weights[index]
            index += 1

        return index - 1

    def next_sample(self, num_samples=1, model_index=None):
        if model_index is None:
            model_index = self.get_weighted_index()
        (means, covariances) = self.models[model_index]
        labels = np.repeat(model_index, num_samples)
        return (np.random.multivariate_normal(means, covariances, num_samples), labels)




