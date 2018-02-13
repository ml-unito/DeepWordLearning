from models.som.SOM import SOM
from utils.utils import load_from_pickle
from utils.constants import Constants
from sklearn.externals import joblib
from .full_length_svc import apply_loaded_pca
import os
import logging
import numpy as np

ACTIVATIONS_PATH = os.path.join(Constants.DATA_FOLDER, 'activations-pca-truncated-80.pkl')

if __name__ == '__main__':
    audio_som = SOM(30, 20, 2000)
    logging.info('Loading pickle')
    activations = load_from_pickle(ACTIVATIONS_PATH)
    activations = [np.ravel(x) for x in activations]
    audio_som.train(activations)
