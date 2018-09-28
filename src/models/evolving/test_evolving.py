from sklearn import datasets
from sklearn import metrics
from sklearn import preprocessing
from models.evolving import EvolvingClustering
import numpy as np

import matplotlib.pyplot as plt
from itertools import cycle, islice

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None
iris = datasets.load_iris()

# X = iris.data
# y = iris.target

X, y = noisy_moons

standardized_X = preprocessing.scale(X)

evol_model = EvolvingClustering.EvolvingClustering(macro_cluster_update=100)
evol_model.fit(standardized_X)

labels = evol_model.labels_

colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(labels) + 1))))
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[labels])
plt.show()
# metrics.silhouette_score(X, labels, metric='euclidean')