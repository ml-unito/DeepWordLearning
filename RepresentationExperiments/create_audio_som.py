from models.som.SOM import SOM
from models.som.SOMTest import showSom
from utils.utils import load_from_pickle
from utils.utils import load_data
from utils.utils import from_csv
from utils.constants import Constants
from sklearn.externals import joblib
from .full_length_svc import apply_loaded_pca
import os
import logging
import numpy as np

ACTIVATIONS_PATH = os.path.join(Constants.DATA_FOLDER, 'activations-10classes.pkl')
csv_path = os.path.join(Constants.DATA_FOLDER, '10classes', 'audio_data.csv')
LOAD = True

if __name__ == '__main__':
    logging.info('Loading data')
    xs, ys = from_csv(csv_path)
    vect_size = len(xs[0])
    audio_som = SOM(20, 30, vect_size,
        checkpoint_dir=os.path.join(Constants.DATA_FOLDER, 'audio_som_10', ''))
    if not LOAD:
        audio_som.train(xs)
    else:
        logging.info('Training som')
        audio_som.restore_trained()
    showSom(audio_som, xs, ys, 1, 'Audio Map')
