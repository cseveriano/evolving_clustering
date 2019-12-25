import numpy as np
import math
from sklearn import preprocessing
from evolving import EvolvingClustering, load_dataset, Metrics, util
from evolving import EvolvingClustering2
import matplotlib.pyplot as plt
import pickle
from matplotlib import cm


def main():

    X, y = load_dataset.load_dataset("s2")
    standardized_X = preprocessing.scale(X)
    minmaxscaler = preprocessing.MinMaxScaler()
    minmaxscaler.fit(standardized_X)
    X = minmaxscaler.transform(standardized_X)

#    evol_model = EvolvingClustering.EvolvingClustering(macro_cluster_update=1, variance_limit=0.01, debug=True)
    evol_model = EvolvingClustering2.EvolvingClustering2(rad=0.04, debug=True)
    evol_model.fit(X[:3000])
    y_pred = evol_model.predict(X[:3000])

if __name__ == '__main__':
    main()