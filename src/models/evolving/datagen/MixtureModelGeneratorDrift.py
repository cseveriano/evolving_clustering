import numpy as np
from evolving.datagen import MixtureModel as mm

class MixtureModelGenerator:

    def __init__(self, **kwargs):

        self.dimensions_pre = kwargs.get('dimensions_pre', 2)
        self.num_models_pre = kwargs.get('num_models_pre', 2)
        self.dimensions_post = kwargs.get('dimensions_post', 2)
        self.num_models_post = kwargs.get('num_models_post', 2)

        self.num_instances_pre_drift = kwargs.get('num_instances_pre_drift', 10)
        self.drift_duration = kwargs.get('drift_duration', 1)
        self.drift_magnitude = kwargs.get('drift_magnitude', 0.5)
        self.drift_precision = kwargs.get('drift_precision', 0.01)

    def prepare_for_use(self):
        self.mixture_model_pre = mm.MixtureModel(self.dimensions_pre, self.num_models_pre)

        self.mixture_model_post = mm.MixtureModel(self.dimensions_post, self.num_models_post, mixture_model_pre=self.mixture_model_pre, target_drift=self.drift_magnitude )


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


## TODO:
# - hellingerDistance
# - next_sample
# concept_drift_test
