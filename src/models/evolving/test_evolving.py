from sklearn import metrics
from sklearn import preprocessing
from models.evolving import EvolvingClustering
import numpy as np
from models.evolving  import load_dataset

import matplotlib.pyplot as plt
from itertools import cycle, islice


X, y = load_dataset.load_dataset("s2")
standardized_X = preprocessing.scale(X)
minmaxscaler = preprocessing.MinMaxScaler()
minmaxscaler.fit(standardized_X)
X = minmaxscaler.transform(standardized_X)

evol_model = EvolvingClustering.EvolvingClustering(macro_cluster_update=1,  variance_limit=0.001, debug=True)
evol_model.fit(X)

labels = evol_model.labels_

colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(labels) + 1))))
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[labels])
plt.show()
# metrics.silhouette_score(X, labels, metric='euclidean')