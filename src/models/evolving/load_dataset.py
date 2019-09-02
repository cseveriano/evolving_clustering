from sklearn import datasets
import numpy as np
import os
import pandas as pd

def load_dataset(dataset_name, n_samples=1500, n_features=2):

    if dataset_name == "moons":
        noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
        X, y = noisy_moons
    elif dataset_name == "noisy_circles":
        noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                              noise=.05)
        X, y = noisy_circles
    elif dataset_name == "blobs":
        blobs = datasets.make_blobs(n_samples=n_samples, n_features=n_features, random_state=8)
        X, y = blobs
    elif dataset_name == "no_structure":
        no_structure = np.random.rand(n_samples, 2), None
        X, y = no_structure
    elif dataset_name == "iris":
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
    elif dataset_name == "stream1":
        path = os.path.join(os.getcwd(), "../../../data/experiments/stream/stream_exp1.csv")
        data = pd.read_csv(path, sep=",")
        X = data[['X1', 'X2']].values
        y = data['class'].values
    elif dataset_name == "gaussian":
        path = os.path.join(os.getcwd(), "../../../data/experiments/stream/gaussian_df.csv")
        data = pd.read_csv(path, sep=",")
        X_columns = data.columns[1:-1]
        X = data[X_columns].values
        y = data['class'].values
    elif dataset_name == "s2":
        Xpath = os.path.join(os.getcwd(), "../../../references/Codigos_MicroTeda/Bases de Dados/s2.txt")
        X = pd.read_csv(Xpath, sep="    ", header=None).values
        ypath = os.path.join(os.getcwd(), "../../../references/Codigos_MicroTeda/Bases de Dados/s2-label.txt")
        y = pd.read_csv(ypath, header=None).values
    elif dataset_name == "kddcup99":
        Xpath = os.path.join(os.getcwd(), "../../../references/Codigos_MicroTeda/Bases de Dados/s2.txt")
        X = pd.read_csv(Xpath, sep="    ", header=None).values
        ypath = os.path.join(os.getcwd(), "../../../references/Codigos_MicroTeda/Bases de Dados/s2-label.txt")
        y = pd.read_csv(ypath, header=None).values
    else:
        print("Dataset not found")

    return X, y