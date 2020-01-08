from sklearn import preprocessing
from evolving import EvolvingClustering, load_dataset, Metrics, util
import matplotlib.pyplot as plt
import pickle
from time import time as time
from benchmarks.denstream.DenStream import DenStream
from benchmarks.clustream.CluStream import CluStream

cmap = plt.cm.get_cmap('rainbow')

#X, y = load_dataset.load_dataset("s2")
#X, y = load_dataset.load_dataset("blobs", n_samples=1000, n_features=2)
X, y = load_dataset.load_dataset("gaussian")

X = X[:100,:20]
standardized_X = preprocessing.scale(X)
minmaxscaler = preprocessing.MinMaxScaler()
minmaxscaler.fit(standardized_X)
X = minmaxscaler.transform(standardized_X)


# CLUSTREAM #########################################

#clustream = CluStream(q=100, m=10, radius_factor = 1.8, delta=10, k=5, init_number=100)
#y_pred = clustream.fit_predict(X)
#y_pred[y_pred == -1] = 5

#print("Purity: %10.4f"% (Metrics.purity(y,y_pred)))
#print("Precision: %10.4f"% (Metrics.precision(y,y_pred)))
#print("Recall: %10.4f"% (Metrics.recall(y,y_pred)))

# CLUSTREAM #########################################

# DENSTREAM #########################################

denstream = DenStream(eps=0.3, lambd=0.1, beta=0.5, mu=3)
#denstream.fit_predict(X)
y_pred = denstream.fit_predict(X)
#y_pred[y_pred == -1] = 5

print("Purity: %10.4f"% (Metrics.purity(y,y_pred)))
print("Precision: %10.4f"% (Metrics.precision(y,y_pred)))
print("Recall: %10.4f"% (Metrics.recall(y,y_pred)))

# DENSTREAM #########################################


# EVOLVING CLUSTERING ###############################
evol_model = EvolvingClustering.EvolvingClustering(variance_limit=0.001, debug=True)

tic = time()
evol_model.fit(X)
tac = time()
print('Operation took {} ms'.format((tac - tic) * 1e3))

y_pred = evol_model.predict(X)

print("Purity: %10.4f"% (Metrics.purity(y,y_pred)))
print("Precision: %10.4f"% (Metrics.precision(y,y_pred)))
print("Recall: %10.4f"% (Metrics.recall(y,y_pred)))

# EVOLVING CLUSTERING ###############################




