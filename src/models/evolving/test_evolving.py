from sklearn import datasets
from sklearn import metrics
from models.evolving import EvolvingClustering

dataset = datasets.load_iris()

X = dataset.data
y = dataset.target

evol_model = EvolvingClustering().fit()

labels = evol_model.labels_
metrics.silhouette_score(X, labels, metric='euclidean')