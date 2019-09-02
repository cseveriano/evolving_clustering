from sklearn import preprocessing
from evolving import EvolvingClustering, load_dataset, Metrics, util
import matplotlib.pyplot as plt
import pickle
from time import time as time

cmap = plt.cm.get_cmap('rainbow')

#X, y = load_dataset.load_dataset("s2")
#X, y = load_dataset.load_dataset("blobs", n_samples=1000, n_features=50)
X, y = load_dataset.load_dataset("gaussian")

standardized_X = preprocessing.scale(X)
minmaxscaler = preprocessing.MinMaxScaler()
minmaxscaler.fit(standardized_X)
X = minmaxscaler.transform(standardized_X)

## Running training and prediction..
evol_model = EvolvingClustering.EvolvingClustering(variance_limit=0.001, debug=True)

tic = time()
evol_model.fit(X)
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

#util.plot_macro_clusters(X, evol_model)

print("Purity: %10.4f"% (Metrics.purity(y,y_pred)))
print("Precision: %10.4f"% (Metrics.precision(y,y_pred)))
print("Recall: %10.4f"% (Metrics.recall(y,y_pred)))
print("CH Score: %10.4f"% (Metrics.ch_score(y,y_pred)))


