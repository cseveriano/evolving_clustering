from sklearn import preprocessing
from evolving import EvolvingClustering2, load_dataset, Metrics, util
import matplotlib.pyplot as plt
import numpy as np
import pickle
from time import time as time
from matplotlib import cm


cmap = plt.cm.get_cmap('rainbow')



X, y = load_dataset.load_dataset("s2")
#X, y = load_dataset.load_dataset("blobs", n_samples=1000, n_features=50)
#X, y = load_dataset.load_dataset("gaussian")

standardized_X = preprocessing.scale(X)
minmaxscaler = preprocessing.MinMaxScaler()
minmaxscaler.fit(standardized_X)
X = minmaxscaler.transform(standardized_X)

## Running training and prediction..
evol_model = EvolvingClustering2.EvolvingClustering2(rad=0.04, debug=True)

tic = time()
evol_model.fit(X[:2000])
tac = time()
print('Operation took {} ms'.format((tac - tic) * 1e3))

##### plot all micro clusters ####

# ax = plt.gca()
# ax.scatter(X[:2000, 0], X[:2000, 1], s=1, color='b')
#
# for i in np.arange(evol_model.micro_obj.nclusters):
#     circle = plt.Circle(evol_model.micro_obj.teda[i].curr_mean, evol_model.micro_obj.raios[i], color='r', fill=False)
#     ax.add_artist(circle)

##### plot all micro clusters ####

##### plot macro clusters ####

ax = plt.gca()
ax.scatter(X[:2000, 0], X[:2000, 1], s=1, color='b')
nmacros = evol_model.macro_obj.macro2.nclust
colors = cm.rainbow(np.linspace(0, 1, nmacros))

for idxs, c in zip(evol_model.macro_obj.macro2.macro_list, colors):
    for idx in idxs:
        circle = plt.Circle(evol_model.micro_obj.teda[idx].curr_mean, evol_model.micro_obj.raios[idx], color=c,
                            fill=False)
        ax.add_artist(circle)

plt.draw()


##### plot macro clusters ####

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

