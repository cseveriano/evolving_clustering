import numpy as np


def prequential_evaluation(method, data, labels, metric, train_size, window_size=1, fading_factor=1):
    '''
    :param method: clustering method to be evaluated
    :param data: data for training and test
    :param labels: ground truth labels
    :param metric: error metric
    :param train_size: number of samples for the first training step
    :param window_size: window size for each prequential iteration
    :param fading_factor: fading factor for older test samples
    :return:
    accumulated error, error list
    '''

    test_end = 0
    index = 0
    limit = len(labels)
    accumulated_error = 0
    error_list = []

    while test_end < limit:
        train_start, train_end, test_start, test_end = getDataIndex(index, train_size, window_size)
        train_data = data[train_start:train_end]
        test_data = data[test_start:test_end]

        method.fit(train_data)
        y_hat = method.predict(test_data)
        y = labels[test_start:test_end]

        error = metric(y, y_hat)
        accumulated_error += fading_factor * error
        error_list.append(error)

    return accumulated_error, error_list


def getDataIndex(index, train_size, window_size):

    train_start = index

    if index == 0:
        train_end = index + train_size
    else:
        train_end = index + window_size

    test_start = train_end
    test_end = test_start + window_size

    return train_start, train_end, test_start, test_end
