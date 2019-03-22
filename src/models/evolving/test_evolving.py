import numpy as np
import math
from sklearn import preprocessing
from evolving import EvolvingClustering, load_dataset, Metrics, util
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle, islice
import pickle
from matplotlib import cm

def plot_macro_clusters(X, model):


    macro_clusters = model.active_macro_clusters
    colors = cm.rainbow(np.linspace(0, 1, len(macro_clusters)))

    ax = plt.gca()

    ax.scatter(X[:, 0], X[:, 1], s=1, color='b')

    for mg, c in zip(macro_clusters, colors):
        for i in mg:
            mi = model.micro_clusters[i]

            mean = mi["mean"]
            std = math.sqrt(mi["variance"])
            circle = plt.Circle(mean, 2 * std, color= c, fill=False)
            ax.add_artist(circle)

    plt.draw()

cmap = plt.cm.get_cmap('rainbow')

X, y = load_dataset.load_dataset("s2")
standardized_X = preprocessing.scale(X)
minmaxscaler = preprocessing.MinMaxScaler()
minmaxscaler.fit(standardized_X)
X = minmaxscaler.transform(standardized_X)

## Running training and prediction..
evol_model = EvolvingClustering.EvolvingClustering(macro_cluster_update=1,  variance_limit=0.01, debug=True)
evol_model.fit(X)
y_pred = evol_model.predict(X)

pickle.dump(evol_model, open("evol_model.pkl", "wb"))
## END Running training and prediction..

## Load pickle
# evol_model = pickle.load(open("evol_model.pkl", "rb"))
# y_pred = evol_model.labels_
## END Load pickle

y_pred = [x+1 for x in y_pred]

# plot_macro_clusters(X, evol_model)
#
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=1, cmap='viridis')
# plt.show()
#
# y = y[:,0]
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, s=1, cmap='viridis')
# plt.show()


# colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
#                                      '#f781bf', '#a65628', '#984ea3',
#                                      '#999999', '#e41a1c', '#dede00']),
#                               int(max(y_pred) + 1))))
#
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
# plt.show()
#
# colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
#                                      '#f781bf', '#a65628', '#984ea3',
#                                      '#999999', '#e41a1c', '#dede00']),
#                               int(max(y) + 1))))
#
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y])
# plt.show()

y_pred = util.adjust_lables(y_pred, y)

print("Purity: %10.4f"% (Metrics.purity(y,y_pred)))
print("Precision: %10.4f"% (Metrics.precision(y,y_pred)))
print("Recall: %10.4f"% (Metrics.recall(y,y_pred)))
print("CH Score: %10.4f"% (Metrics.ch_score(y,y_pred)))


