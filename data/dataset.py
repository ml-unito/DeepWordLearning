import os, sys, logging, glob, librosa
import numpy as np
import pickle
from utils.constants import Constants
logging.basicConfig(level=Constants.LOGGING_LEVEL)

class Dataset():

    def __init__(self, root_folder):
        self.root_folder = root_folder

    def load():
        raise NotImplementedError('This is an abstract class. \n Load the \
                                   Dataset using a subclass.')

    def __add__(self, other):
        temp_data = Dataset(self.root_folder + ':' + other.root_folder)
        temp_data.X = self.X + other.X
        temp_data.y = self.y + other.y
        temp_data.loaded = True
        return temp_data

    def to_file(self):
        filename = ""
        names = self.root_folder.split(':')
        i = 0
        for name in names:
            last_dir_name = name.split(os.path.sep)[-1]
            filename = filename + last_dir_name
            if i+1 != len(names):
                filename += '-'
            i += 1
        filename += ".pickle"
        with open(filename, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def __radd__(self, other):
        return self.__add__(other)


class OSXSpeakerDataset(Dataset):

    def __init__(self, speaker_name):
        if speaker_name in Constants.AVAILABLE_SPEAKERS:
            self.root_folder = os.path.join(Constants.AUDIO_DATA_FOLDER, speaker_name)
            self.loaded = False
        else:
            print('Unsupported speaker name')
            sys.exit(1)

    def load(self):
        if self.loaded == True:
            logging.error('This Dataset instance has been loaded already!')
            return
        folder_list = glob.glob(os.path.join(self.root_folder, '*'))
        logging.debug('Found ' + str(len(folder_list)) + ' subfolders.')
        X = []
        y = []
        for folder in folder_list:
            temp_y = folder.split(os.path.sep)[-1]
            first_file = glob.glob(os.path.join(folder, '*'))[0]
            temp_X, _ = librosa.core.load(first_file) # drop sampling rate info
            X.append(temp_X)
            y.append(temp_y)
        logging.debug('Dataset X loaded with shape ' + str(np.shape(X)))
        logging.debug('Dataset y loaded with shape ' + str(np.shape(y)))

        self.X = X
        self.y = y
        self.loaded = True


    def pad_dataset(self, X):
        ''' Pads a list of numpy arrays. The one with maximum length will force the others to its length.'''
        logging.debug('X[0] shape before padding: ' + str(np.shape(X[0])))
        max_length = 0
        for x in X:
            if np.shape(x)[0] > max_length:
                max_length = np.shape(x)[0]
        X_padded = []
        for x in X:
            x_temp = np.lib.pad(x, (0, max_length - np.shape(x)[0]), 'constant', constant_values=(None, 0))
            X_padded.append(x_temp)
        self.max_length = max_length
        self.X = X_padded
        logging.debug('X[0] shape after padding: ' + str(np.shape(X[0])))

if __name__ == '__main__':
    c = OSXSpeakerDataset('tom')
    c.load()
