from sklearn import preprocessing
from evolving import EvolvingClustering, load_dataset, Metrics, util, Benchmarks
import matplotlib.pyplot as plt
import numpy as np


X, y = load_dataset.load_dataset("s2")
standardized_X = preprocessing.scale(X)
minmaxscaler = preprocessing.MinMaxScaler()
minmaxscaler.fit(standardized_X)
X = minmaxscaler.transform(standardized_X)
y = np.array([el[0] for el in y])

evol_model = EvolvingClustering.EvolvingClustering(macro_cluster_update=1,  variance_limit=0.01, debug=True)

train_size = 100
window_size = 100
result = Benchmarks.prequential_evaluation(evol_model, X[:500], y[:500], Metrics.precision,train_size, window_size, elapsed_time=True)

plt.figure()
plt.plot(result['error_list'], np.arange(len(result['error_list'])))
