from .data_utils import load_data
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC
import pickle
from utils.constants import Constants
import os
import logging

TEST = False
MAX_LEN = 80

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

def apply_loaded_pca(xs, path):
    pca = joblib.load(path)
    print('xs[0]: {}'.format(str(xs[0].shape)))
    xs = np.array([pca.transform(x) for x in xs])
    print('xs[0]: {}'.format(str(xs[0].shape)))
    return xs

def train_svc(xs, ys, k = 5):
    kf = KFold(n_splits=5)
    results_rbf = []
    results_linear = []
    flat_xs = np.array([x.ravel() for x in xs])
    print([x.shape for x in flat_xs][0:50])
    ys = np.array(ys)
    for train_i, test_i in kf.split(flat_xs):
        rbf = SVC()
        linear = LinearSVC()
        print('Fitting RBF...')
        rbf.fit(flat_xs[train_i], ys[train_i])
        print('Fitting linear...')
        linear.fit(flat_xs[train_i], ys[train_i])
        pred = rbf.predict(flat_xs[test_i])
        results_rbf.append(np.average(pred == ys[test_index]))
        pred = linear.predict(flat_xs[test_i])
        results_linear.append(np.average(pred == ys[test_index]))
    print('SVC RBF: {}; SVC Linear: {}'.format(np.average(results_rbf), np.average(results_linear)))
        
if __name__ == '__main__':
    logging.info('Loading pickle')
    xs, ys = load_data(os.path.join(Constants.DATA_FOLDER, 'activations.pkl'))
    #max_len = np.max([x.shape[0] for x in xs])
    logging.info('Applying PCA')
    xs = apply_loaded_pca(xs, os.path.join(Constants.DATA_FOLDER, 'pca_25.pkl'))
    logging.info('Fixing sequence length')
    xs = fix_seq_length(xs, length=MAX_LEN)
    logging.info('Dumping pickle')
    with open(os.path.join(Constants.DATA_FOLDER, 'activations-pca-truncated-'+str(MAX_LEN)+'.pkl'), 'wb') as f:
        pickle.dump(xs, f)
    if TEST == True:
        train_svc(xs, ys)
    
