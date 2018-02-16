from models.som.SOM import SOM
from models.som.SOMTest import showSom
from utils.utils import load_from_pickle
from utils.utils import load_data
from utils.constants import Constants
from sklearn.externals import joblib
from .full_length_svc import apply_loaded_pca
import os
import logging
import numpy as np

ACTIVATIONS_PATH = os.path.join(Constants.DATA_FOLDER, 'activations-10classes.pkl')
csv_path = os.path.join(Constants.DATA_FOLDER, 'audio-training-10classes.csv')
LOAD = False

if __name__ == '__main__':
    audio_som = SOM(20, 30, 2000,
        checkpoint_dir=os.path.join(Constants.DATA_FOLDER, 'audio_som_10', ''))
    logging.info('Loading pickle')
    xs, ys = load_data(ACTIVATIONS_PATH)
    xs = [np.ravel(x) for x in xs]
    logging.info('Training som')
    if not LOAD:
        audio_som.train(xs)
    else:
        audio_som.restore_trained()
    showSom(audio_som, xs, ys, 1, 'Audio Map')
