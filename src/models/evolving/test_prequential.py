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

train_size = 300
window_size = 100
accum_error, error_list = Benchmarks.prequential_evaluation(evol_model, X[0:400], y[0:400], Metrics.precision,train_size, window_size)

plt.figure()
plt.plot(error_list, np.arange(len(error_list)))
