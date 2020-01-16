from sklearn import preprocessing
from evolving import EvolvingClustering, util
from evolving.util import Metrics, load_dataset
import matplotlib.pyplot as plt
from time import time as time

cmap = plt.cm.get_cmap('rainbow')

#X, y = load_dataset.load_dataset("s2")
#X, y = load_dataset.load_dataset("blobs", n_samples=1000, n_features=50)
X, y = load_dataset.load_dataset("gaussian")

nchunks = 500
X = X[:6000,:2]
y = y[:6000]
standardized_X = preprocessing.scale(X)
minmaxscaler = preprocessing.MinMaxScaler()
minmaxscaler.fit(standardized_X)
X = minmaxscaler.transform(standardized_X)

#util.plot_data_labels(X, y)

## Running training and prediction..
evol_model = EvolvingClustering.EvolvingClustering(variance_limit=0.001, decay=1000, debug=True)

tic = time()

evol_model.fit(X)

# ind = 0
# for chunk in np.array_split(X, nchunks):
#     ind += len(chunk)
#     evol_model.fit(chunk)
#     util.plot_macro_clusters(X[:ind], evol_model)

tac = time()
print('Operation took {} ms'.format((tac - tic) * 1e3))

y_pred = evol_model.predict(X)

#pickle.dump(evol_model, open("evol_model.pkl", "wb"))
## END Running training and prediction..

## Load pickle
# evol_model = pickle.load(open("evol_model.pkl", "rb"))
# y_pred = evol_model.labels_
## END Load pickle

#y_pred = [x+1 for x in y_pred]

#y_pred = util.adjust_labels(y_pred, y)


util.plot_macro_clusters(X, evol_model)

# print("Purity: %10.4f"% (Metrics.purity(y,y_pred)))
print("Precision: %10.4f" % (Metrics.precision(y, y_pred)))
# print("Recall: %10.4f"% (Metrics.recall(y,y_pred)))
# print("CH Score: %10.4f"% (Metrics.ch_score(y,y_pred)))


