import numpy as np
from sklearn.metrics import cluster, precision_score, recall_score, calinski_harabaz_score

def purity(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)

    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='micro')


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='micro')


def ch_score(X, y_pred):
    return calinski_harabaz_score(X, y_pred)

