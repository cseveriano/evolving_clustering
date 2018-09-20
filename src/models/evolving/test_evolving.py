from sklearn import datasets
from sklearn import metrics
from models.evolving import EvolvingClustering
import numpy as np

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

evol_model = EvolvingClustering.EvolvingClustering()
evol_model.fit(X)

labels = evol_model.labels_
metrics.silhouette_score(X, labels, metric='euclidean')