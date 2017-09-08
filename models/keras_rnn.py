import librosa
import numpy as np
import logging
from functools import reduce
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.utils import to_categorical
from data.dataset import OSXSpeakerDataset
from sklearn.model_selection import train_test_split
from utils.constants import Constants


def build_keras_rnn():
    model = Sequential()
    model.add(SimpleRNN(100, input_shape=(30000, 1), return_sequences = True))
    model.add(SimpleRNN(100, return_sequences = False))
    model.add(Dense(1000, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def train_keras_rnn(model):
    # load prepare the data
    train_speakers_dataset, test_speaker_dataset = load_all_speakers()
    X_train = train_speakers_dataset.X
    y_train = train_speakers_dataset.y
    X_test = test_speaker_dataset.X
    y_test = test_speaker_dataset.y
    logging.debug('Total size of train dataset: ' + str(np.shape(X_train)))
    logging.debug('Total size of test dataset: ' + str(np.shape(X_test)))

    # pad sequences
    maximum_length = 30000
    X_train = sequence.pad_sequences(X_train, maxlen=maximum_length)
    X_test = sequence.pad_sequences(X_test, maxlen=maximum_length)

    X_train = np.reshape(X_train, (X_train.shape[0], maximum_length, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], maximum_length, 1))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=32)
    return model

def test_keras_rnn():
    pass

def load_all_speakers():
    accumulator_dataset = None
    i = 0
    dataset_list = []
    for speaker in Constants.AVAILABLE_SPEAKERS:
        temp_dataset = OSXSpeakerDataset(speaker)
        temp_dataset.load()
        dataset_list.append(temp_dataset)
    return reduce((lambda x, y: x + y), dataset_list[:-1]), dataset_list[-1]


if __name__ == '__main__':
    np.random.seed(7)
    model = build_keras_rnn()
    trained_model = train_keras_rnn(model)
    test_keras_rnn(trained_model)
