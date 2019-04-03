from sklearn import preprocessing
from evolving import EvolvingClustering, load_dataset, Metrics, util, Benchmarks
import matplotlib.pyplot as plt
import numpy as np

X, y = load_dataset.load_dataset("stream1")
# standardized_X = preprocessing.scale(X)
# minmaxscaler = preprocessing.MinMaxScaler()
# minmaxscaler.fit(standardized_X)
# X = minmaxscaler.transform(standardized_X)
# y = np.array([el[0] for el in y])

evol_model = EvolvingClustering.EvolvingClustering(macro_cluster_update=1,  variance_limit=0.01, debug=True)

# Experiment parameters
nclusters = 4
nsamples = 2000 * nclusters
train_size = 800 * nclusters
window_size = 100

# evol_model = EvolvingClustering.EvolvingClustering(macro_cluster_update=1,  variance_limit=0.001, debug=True)
# evol_model.fit(X[:train_size])
# util.plot_macro_clusters(X, evol_model)
#
#
# y_pred = evol_model.predict(X)
#
# y_pred = util.adjust_labels(y_pred, y)

accum_error, error_list = Benchmarks.prequential_evaluation(evol_model, X, y, Metrics.precision,train_size, window_size, adjust_labels=2)

plt.figure()
plt.plot(error_list, np.arange(len(error_list)))
