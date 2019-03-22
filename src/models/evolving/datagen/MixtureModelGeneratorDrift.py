import numpy as np
from evolving.datagen import MixtureModel as mm

class MixtureModelGeneratorDrift:

    def __init__(self, **kwargs):

        self.dimensions = kwargs.get('dimensions', 2)
        self.num_models_pre = kwargs.get('num_models_pre', 2)
        self.num_models_post = kwargs.get('num_models_post', 2)

        self.num_instances_pre_drift = kwargs.get('num_instances_pre_drift', 10)
        self.drift_duration = kwargs.get('drift_duration', 1)
        self.drift_magnitude = kwargs.get('drift_magnitude', 0.5)
        self.num_samples = 0

    def prepare_for_use(self):
        self.mixture_model_pre = mm.MixtureModel(self.dimensions, self.num_models_pre)

        self.mixture_model_post = mm.MixtureModel(self.dimensions, self.num_models_post, mixture_model_pre=self.mixture_model_pre, target_drift=self.drift_magnitude)


    def get_weighted_index(self):
        weights  = self.mixture_model_pre.get_weights()
        probSum = np.sum(weights)
        threshold = np.random.rand() * probSum
        index = 0
        partial = 0.0

        while partial <= threshold and index < len(weights):
            partial += weights[index]
            index += 1

        return index - 1

    def get_random_model_index(self, num_models):
        return np.random.choice(np.arange(num_models))

    def next_sample(self, num_samples=1, model_index=None):

        samples = []
        labels = []
        for i in np.arange(num_samples):
            mixmodel = self.get_current_mixture_model()

            if model_index is None:
                index = self.get_weighted_index()
            else:
                index = model_index

            mean = mixmodel.get_mean(index)
            covariance = mixmodel.get_covariance(index)
            samples.append(np.random.multivariate_normal(mean, covariance))
            labels.append(index)

        return np.array(samples), np.array(labels)

    def get_current_mixture_model(self):
        mixture_model = None
        if self.num_samples <= self.num_instances_pre_drift:
            mixture_model = self.mixture_model_pre
        elif self.num_samples <= (self.num_instances_pre_drift + self.drift_duration):
            if np.random() < ((self.num_samples - self.num_instances_pre_drift) / self.drift_duration):
                mixture_model = self.mixture_model_post
            else:
                mixture_model = self.mixture_model_pre
        elif self.num_samples > (self.num_instances_pre_drift + self.drift_duration):
            mixture_model = self.mixture_model_post

        self.num_samples += 1
        return mixture_model