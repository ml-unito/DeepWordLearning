import numpy as np
import logging
import random
import pickle
import re
import csv

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

def load_from_pickle(path):
    with open(path, 'rb') as f:
        activations = pickle.load(f)
    return activations

def extract_key(keyname):
    extract_key.re = re.compile(r".*/(\d+)")
    match = extract_key.re.match(keyname)
    if match == None:
        raise "Cannot match a label in key: %s" % (keyname)
    return match.group(1)

def load_data(filename):
    """
    Loads the data from filename and parses the keys inside
    it to retrieve the labels. Returns a pair xs,ys representing
    the data and the labels respectively
    """
    data = None
    with (open(filename, "rb")) as file:
        data = pickle.load(file)

    xs = []
    ys = []

    for key in data.keys():
        xs.append(data[key])
        #ys.append(extract_key(key))
        ys.append(key)

    return (xs,ys)

def to_csv(xs, ys, path):
    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for x, y in zip(xs, ys):
            x = np.ravel(x)
            string = np.array2string(x, separator=',').strip('[]')
            string += ',' + y + '\n'
        writer.writerow(string)
