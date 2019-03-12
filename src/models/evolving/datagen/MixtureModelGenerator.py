import numpy as np
from evolving.datagen import MixtureModel as mm

class MixtureModelGenerator:

    def __init__(self, dimensions, num_models):

        self.dimensions = dimensions
        self.num_models = num_models


    def prepare_for_use(self):
        self.mixture_model = mm.MixtureModel(self.dimensions, self.num_models)


    def get_weighted_index(self):
        weights  = self.mixture_model.get_weights()
        probSum = np.sum(weights)
        threshold = np.random.rand() * probSum
        index = 0
        partial = 0.0

        while partial <= threshold and index < len(weights):
            partial += weights[index]
            index += 1

        return index - 1

    def next_sample(self, num_samples=1, model_index=None):
        if model_index is None:
            model_index = self.get_weighted_index()

        mean = self.mixture_model.get_mean(model_index)
        covariance = self.mixture_model.get_covariance(model_index)
        label = np.repeat(model_index, num_samples)

        return (np.random.multivariate_normal(mean, covariance, num_samples), label)