from sklearn import preprocessing
from evolving import EvolvingClustering
from evolving.util import Metrics, Benchmarks, load_dataset
import matplotlib.pyplot as plt
import numpy as np

cmap = plt.cm.get_cmap('rainbow')

#X, y = load_dataset.load_dataset("s2")
#X, y = load_dataset.load_dataset("blobs", n_samples=1000, n_features=50)
X, y = load_dataset.load_dataset("gaussian")

standardized_X = preprocessing.scale(X)
minmaxscaler = preprocessing.MinMaxScaler()
minmaxscaler.fit(standardized_X)
X = minmaxscaler.transform(standardized_X)
y = np.array([el[0] for el in y])

evol_model = EvolvingClustering.EvolvingClustering(variance_limit=0.001, debug=True)

train_size = 100
window_size = 100
result = Benchmarks.prequential_evaluation(evol_model, X[:500], y[:500], Metrics.precision, train_size, window_size, elapsed_time=True)


