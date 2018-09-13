import math
import numpy as np

class EvolvingClustering:
    def __init__(self, max_iter=300, tol=1e-4,
                 verbose=0, variance_limit=0.001):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.total_num_samples = 0
        self.micro_groups = []
        self.variance_limit = variance_limit

    @staticmethod
    def get_micro_group(num_samples, mean, variance):
        return {"num_samples": num_samples, "mean": mean, "variance": variance}

    def create_new_micro_group(self, x):

        num_samples = 1
        mean = x
        variance = 0

        self.micro_groups.append(EvolvingClustering.get_micro_group(num_samples, mean, variance))

    def is_outlier(self, x, micro_group):

        s_ik = micro_group["num_samples"]
        mu_ik = micro_group["mean"]
        var_ik = micro_group["variance"]

        mik_sik = 3 / (1 + math.exp(-0.007 * (s_ik - 100)))
        outlier_limit = (mik_sik ** 2) + 1 / (2 * s_ik)
        norm_ecc = EvolvingClustering.get_normalized_eccentricity(x, s_ik, mu_ik, var_ik)

        cond1 = (norm_ecc > outlier_limit)

        if s_ik == 2:
            cond2 = ((var_ik ** 2) > self.variance_limit)
        else:
            cond2 = True

        return cond1 and cond2

    def update_micro_group(self, x, micro_group):
        s_ik = micro_group["num_samples"]
        mu_ik = micro_group["mean"]
        var_ik = micro_group["variance"]

        s_ik += 1
        mean = ((s_ik -1) / s_ik) * mu_ik + (x / s_ik)
        a = mean - x
        variance = ((s_ik -1) / s_ik) * var_ik + (1 / (s_ik - 1)) * np.dot(a,a)

        micro_group["num_samples"] = s_ik
        micro_group["mean"] = mean
        micro_group["variance"] = variance

    @staticmethod
    def get_normalized_eccentricity(x, num_samples, mean, var):
        return EvolvingClustering.get_eccentricity(x, num_samples, mean, var) / 2

    @staticmethod
    def get_eccentricity(x, num_samples, mean, var):
        a = mean - x
        result = ((1/num_samples) + (np.dot(a, a) / (num_samples * (var))))
        return result

    def fit(self, X):

        for xk in X:

            # First sample
            if self.total_num_samples == 0:
                self.create_first_micro_group(xk)

            else:
                new_micro_group = True

                for mi in self.micro_groups:

                    if not self.is_outlier(xk, mi):
                        self.update_microgroup(xk, mi)
                        new_micro_group = False

                if new_micro_group:
                    self.create_new_micro_group(xk)

                self.total_num_samples += 1

        return self

    def partial_fit(self, X, y=None):
        return self

    def predict(self, X):
        labels = []

        return labels