import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm

def adjust_labels(y_pred, y):
    new_y_pred = np.array(y_pred.copy())

    pred_labels = np.unique(y_pred)

    for l in pred_labels:
        labels = y[y_pred == l]

        uniqueValues, occurCount = np.unique(labels, return_counts=True)

        new_label = uniqueValues[np.argmax(occurCount)]
        new_y_pred[y_pred == l] = new_label

    return new_y_pred.tolist()

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

