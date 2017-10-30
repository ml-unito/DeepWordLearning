import os, sys, logging, glob, librosa
import numpy as np
import pickle
import datetime
import tensorflow as tf
from utils.constants import Constants
logging.basicConfig(level=Constants.LOGGING_LEVEL)

class Dataset():

    def __init__(self, root_folder):
        self.root_folder = root_folder

    def load(self):
        raise NotImplementedError('This is an abstract class. \n Load the \
                                   Dataset using a subclass.')

    def __add__(self, other):
        temp_data = Dataset(self.root_folder + ':' + other.root_folder)
        temp_data.X = self.X + other.X
        temp_data.y = self.y + other.y
        temp_data.loaded = True
        return temp_data

    def __radd__(self, other):
        return self.__add__(other)

    def to_file(self):
        raise NotImplementedError('This is an abstract class. \n Dump the \
                                   Dataset to file using a subclass.')

    def get_placeholders(self, batch_size, sparse=False):
        x_placeholder = tf.placeholder(tf.float32, shape=(batch_size, None, self.X_train.shape[1]))
        if sparse:
            y_placeholder = tf.sparse_placeholder(tf.int32, shape=(batch_size))
        else:
            y_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
        return x_placeholder, y_placeholder

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

class TIMITDataset(Dataset):
    # phoneme dictionary
    phoneme_dict = Constants.TIMIT_PHONEME_DICT
    signal_rate = 16000

    def __init__(self):
        self.root_folder = os.path.join(Constants.TIMIT_DATA_FOLDER)
        self.loaded = False

    def load(self, get_mfcc=True, n_mfcc=13):
        if self.loaded == True:
            logging.error('This Dataset instance has been loaded already!')
            return
        train_folder = os.path.join(self.root_folder, 'TRAIN')
        test_folder = os.path.join(self.root_folder, 'TEST')
        self.X_train, self.y_train = TIMITDataset.load_explore_timit(train_folder, get_mfcc, n_mfcc)
        self.X_test, self.y_test = TIMITDataset.load_explore_timit(test_folder, get_mfcc, n_mfcc)
        self.loaded = True
        self.has_mfcc = get_mfcc

    def to_file(self):
        filename = "timit_"
        date_object = datetime.date.today()
        filename = filename + str(date_object.year) + str(date_object.month) + str(date_object.day)
        if self.has_mfcc:
            filename += "_mfcc"
        filename += ".pickle"
        with open(filename, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @staticmethod
    def load_explore_timit(folder, get_mfcc, n_mfcc, frame_length_seconds=0.010, frame_step_seconds=0.005):
        X = []
        y = []
        for dialect_subfolder in glob.glob(os.path.join(folder, '*')):
            for speaker_subsubfolder in glob.glob(os.path.join(dialect_subfolder, '*')):
                logging.info('Loading ' + str(speaker_subsubfolder) + '...')

                audio_file_list = []
                for audio_file in glob.glob(os.path.join(speaker_subsubfolder, '*.WAV')):
                    audio_file_list.append(audio_file)
                audio_file_list.sort()

                phonem_file_list = []
                for phonem_file in glob.glob(os.path.join(speaker_subsubfolder, '*.PHN')):
                    phonem_file_list.append(phonem_file)
                phonem_file_list.sort()

                dataset_tuple_list = [(af, pf) for af, pf in zip(audio_file_list, phonem_file_list)]

                for audio_file, phonem_file in dataset_tuple_list:
                    temp_X, sr = librosa.core.load(audio_file, sr=16000)
                    if get_mfcc == True:
                        temp_X = TIMITDataset.get_mfcc_from_audio(temp_X, sr, n_mfcc, frame_length_seconds, frame_step_seconds)
                    X.append(temp_X)

                    y_temp = []
                    with open(phonem_file, 'r') as phonetic_transcription_file:
                        temp_y = phonetic_transcription_file.read()
                        temp_y = TIMITDataset.parse_phoneme_string(temp_y, frame_step_seconds*TIMITDataset.signal_rate)

                    # pad the phonetic transcription, if needed
                    temp_y += [0 for i in range(len(temp_X) - len(temp_y))]
                    y.append(temp_y)
                    assert len(temp_X) == len(temp_y)
        return np.array(X), np.array(y)

    @staticmethod
    def get_mfcc_from_audio(audio, signal_rate, n_mfcc, frame_length_seconds, frame_step_seconds):
        mfcc = librosa.feature.mfcc(y=audio, sr=signal_rate, n_mfcc=n_mfcc,
                n_fft=int(frame_length_seconds*signal_rate),
                hop_length=int(frame_step_seconds*signal_rate))
        # add energy info to feature array
        energy = librosa.feature.rmse(audio, n_fft=int(frame_length_seconds*signal_rate), hop_length=int(frame_step_seconds*signal_rate))
        np.insert(mfcc, 0, energy, axis=0)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        temp_X = np.concatenate((mfcc, mfcc_delta, mfcc_delta2)).T # so that each row is a time step
        return temp_X

    @staticmethod
    def parse_phoneme_string(string, divisor):
        lines = string.split('\n')[:-1] # last line in .phn files is always empty for some reason
        temp_list = []
        old_end_time = 0
        for line in lines:
            tokens = line.split(' ')
            start_time = int(tokens[0]) // divisor
            end_time = int(tokens[1]) // divisor
            # sometimes the fft window is too small for the phonem, so start_time and end_time
            # will be the same. in this situation, we drop the label. this happens with a frequency
            # p=10^-5 using the default parameters
            try:
                assert end_time != start_time
            except AssertionError:
                continue
            old_end_time = end_time
            label = TIMITDataset.phoneme_dict[tokens[2]]
            temp_list += [label] * int(end_time - start_time)
        return temp_list

if __name__ == '__main__':
    c = OSXSpeakerDataset('tom')
    c.load()
