from models.som.SOM import SOM
from models.som.wordLearningTest import iterativeTraining
from utils.constants import Constants
from utils.utils import from_csv_with_filenames
import os


"""
Train an auditive som, test it alongside the visual one
"""

somv_path = os.path.join(Constants.DATA_FOLDER,
                         '10classes',
                         'visual_model')

somu_path = os.path.join(Constants.DATA_FOLDER,
                         '10classes',
                         'audio_model')

audio_data_path = os.path.join(Constants.DATA_FOLDER,
                               '10classes',
                               'audio_data_40t.csv')

if __name__ == '__main__':
    xs, ys, filenames = from_csv_with_filenames(audio_data_path)
    vect_size = len(xs[0])
    audio_som = SOM(20, 30, vect_size, n_iterations=100,
        checkpoint_dir=somu_path)
    audio_som.train(xs)
    iterativeTraining(somv_path, somu_path)
