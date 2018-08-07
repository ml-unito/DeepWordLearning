import numpy as np
import logging
import random
import pickle
import re
import csv
import sys
import json
import os
import datetime
from sklearn.decomposition import PCA
from utils.constants import Constants

def fix_seq_length(xs, length=50):
    truncated = 0
    padded = 0
    print('xs[0]: {}'.format(str(xs[0].shape)))
    for i, x in enumerate(xs):
        if length < x.shape[0]:
            x = x[0:length][:]
            truncated += 1
        elif length > x.shape[0]:
            x = np.pad(x, ((0, length - x.shape[0]), (0, 0)), mode='constant', constant_values=(0))
            padded += 1
        xs[i] = x
    print('xs[0]: {}'.format(str(xs[0].shape)))
    print('Truncated {}; Padded {}'.format(truncated/len(xs), padded/len(xs)))
    return xs

def apply_pca(xs, n_components=25):
    pca = PCA(n_components=n_components)
    pca.fit([xi for x in xs for xi in x])
    print('xs[0]: {}'.format(str(xs[0].shape)))
    xs = [pca.transform(x) for x in xs]
    print('xs[0]: {}'.format(str(xs[0].shape)))
    return xs

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

def to_csv(xs, ys, path, filename_list=None):
    np.set_printoptions(threshold=np.inf)
    with open(path, 'w') as csvfile:
        i = 0
        for x, y in zip(xs, ys):
            string = ""
            for xi in x:
                string = string + str(xi) + ','
            if filename_list != None:
                string = filename_list[i] + ',' + string
            string = string + str(y) + '\n'
            csvfile.write(string)
            i += 1
    np.set_printoptions(threshold=1000)

def from_csv(path):
    xs = []
    ys = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x = np.asfarray(row[:-1])
            y = row[-1]
            xs.append(x)
            ys.append(y)
    return xs, ys

def from_csv_with_filenames(path):
    xs = []
    ys = []
    filenames = []
    with open(path, newline='') as csvfile:
        for line in csvfile:
            l = line.split(',')
            xs.append(np.asfarray(l[1:-1]))
            ys.append(l[-1].strip('\n'))
            filenames.append(l[0])
    return xs, ys, filenames

def infer_label_10classes(label_string, labels_dict):
    label_string = label_string.split('/')[5]
    return labels_dict[label_string]

def from_csv_visual_10classes(path):
    f = open(path,'r')
    labels_dict_path = os.path.join(Constants.DATA_FOLDER, 'imagenet-labels.json')
    labels_dict = json.load(open(labels_dict_path))
    labels_dict = {v: k for k, v in labels_dict.items()}
    xs = []
    ys = []
    for l in f:
        lSplit = l.split(',')
        xs.append(np.array(lSplit[1:]).astype(float))
        ys.append(infer_label_10classes(lSplit[0], labels_dict))
    f.close()
    return xs, ys

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def get_plot_filename(folder_path):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    i = 0
    while os.path.exists(os.path.join(folder_path, date_str + "_" + str(i)) + '.png'):
        i += 1
    return date_str + "_" + str(i) + '.png'
