from sklearn import preprocessing
from evolving import EvolvingClustering, load_dataset, Metrics, util, Benchmarks
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

X, y = load_dataset.load_dataset("s2")
standardized_X = preprocessing.scale(X)
minmaxscaler = preprocessing.MinMaxScaler()
minmaxscaler.fit(standardized_X)
X = minmaxscaler.transform(standardized_X)
y = np.array([el[0] for el in y])

evol_model = EvolvingClustering.EvolvingClustering(macro_cluster_update=1,  variance_limit=0.01, debug=False)

train_size = 3000
window_size = 100
Benchmarks.monte_carlo_evaluation(evol_model, adjusted_rand_score, X[0:400], y[0:400],  trials=10)
