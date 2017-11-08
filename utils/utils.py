import numpy as np
import logging
import random

def pad_np_arrays(X):
    ''' Pads a list of numpy arrays. The one with maximum length will force the others to its length.'''
    logging.debug('Shape of first example before padding: ' + str(X[0].shape))
    logging.debug('Shape of dataset before padding: ' + str(np.shape(X)))

    max_length = 0
    for x in X:
        if np.shape(x)[0] > max_length:
            max_length = np.shape(x)[0]
    X_padded = []
    for x in X:
        x_temp = np.lib.pad(x, (0, max_length - np.shape(x)[0]), 'constant', constant_values=(None, 0))
        X_padded.append(x_temp)
    logging.debug('Shape of first example after padding: ' + str(X_padded[0].shape))
    logging.debug('Shape of dataset after padding: ' + str(np.shape(X_padded)))
    return X_padded

def array_to_sparse_tuple(X):
    indices = []
    values = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            indices.append([i, j])
            values.append(X[i, j])
    return indices, values

def array_to_sparse_tuple_1d(X):
    indices = []
    values = []
    for i in range(X.shape[0]):
        indices.append(i)
        values.append(X[i])
    return indices, values

def get_next_batch_index(possible_list):
    i = random.randrange(0, len(possible_list))
    return possible_list[i]
