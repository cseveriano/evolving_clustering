from sklearn import preprocessing
from evolving import EvolvingClustering, util
from evolving.util import Metrics, Benchmarks, load_dataset
import matplotlib.pyplot as plt
import numpy as np


X, y = load_dataset.load_dataset("gaussian")
standardized_X = preprocessing.scale(X)
minmaxscaler = preprocessing.MinMaxScaler()
minmaxscaler.fit(standardized_X)
X = minmaxscaler.transform(standardized_X)
#y = np.array([el[0] for el in y])

evol_model = EvolvingClustering.EvolvingClustering(variance_limit=0.001, decay=1000, debug=True)

nsamples = 6000
train_size = 1000
window_size = 1000
X = X[:nsamples,:2]
y = y[:nsamples]

result = Benchmarks.prequential_evaluation(evol_model, X, y, Metrics.precision, train_size, window_size, elapsed_time=True)

util.plot_macro_clusters(X, evol_model)

fig = plt.figure(figsize=(14,6))

windows = np.arange(train_size+window_size,nsamples+window_size,window_size)
plt.plot(windows,result['error_list'],'o-', color='blue',label='Evolving')