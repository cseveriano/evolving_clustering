import numpy as np

def adjust_lables(y_pred, y):
    new_y_pred = np.array(y_pred.copy())

    pred_labels = np.unique(y_pred)

    for l in pred_labels:
        labels = y[y_pred == l]

        uniqueValues, occurCount = np.unique(labels, return_counts=True)

        new_label = uniqueValues[np.argmax(occurCount)]
        new_y_pred[y_pred == l] = new_label

    return new_y_pred.tolist()

