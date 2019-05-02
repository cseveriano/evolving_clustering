from sklearn.metrics import adjusted_rand_score
from evolving import EvolvingClustering, load_dataset, Metrics, util, Benchmarks
import matplotlib.pyplot as plt
import numpy as np

def main():
    X, y = load_dataset.load_dataset("stream1")

    # Experiment parameters
    nclusters = 4
    nsamples = 2000 * nclusters
    train_size = 800 * nclusters
    window_size = 100

    evol_model = EvolvingClustering.EvolvingClustering(macro_cluster_update=1, variance_limit=0.001, debug=True)
    Benchmarks.prequential_evaluation(evol_model, X, y, adjusted_rand_score, train_size,
                                                                window_size)

if __name__ == '__main__':
    main()