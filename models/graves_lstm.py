import tensorflow as tf
import librosa as lr
import pickle
import glob
import numpy as np
from data.dataset import TIMITDataset
from keras.layers import Input, Bidirectional, LSTM
from keras.models import Sequential, Model


LOAD_PICKLE = False
NUM_LAYERS = 2
NUM_CLASSES = 10 # placeholder
NUM_HIDDEN = 50
NUM_FEATURES = 20 # placeholder
NUM_TIMESTEPS = 100 # placeholder

def inference():
    pass

def loss():
    pass

def training():
    pass

def create_model(dataset):
    num_layers = NUM_LAYERS
    num_features = dataset.X_train.shape[1]
    x, y = dataset.get_placeholders()

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    cell = tf.contrib.rnn.LSTMCell(NUM_HIDDEN, state_is_tuple=True)
    stacked_layers = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    outputs, _ = tf.nn.dynamic_rnn(stacked_layers, x, seq_len, dtype=tf.float32)

    shape = tf.shape(x)
    batch_s, max_time_steps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, NUM_CLASSES])

    pass

def ctc_loss(y_true,y_pred):
        return(tf.nn.ctc_loss(y_pred, y_true, 64,
                                       preprocess_collapse_repeated=False, ctc_merge_repeated=False,
                                       time_major=True))

def create_model_keras():
    x = Input(shape=(NUM_FEATURES,NUM_TIMESTEPS, None))
    y_pred = Bidirectional(LSTM(NUM_HIDDEN, return_sequences=True), merge_mode='sum')(x)
    #for i in range(0, NUM_LAYERS-2):
    #    y_pred = Bidirectional(LSTM(NUM_HIDDEN, return_sequences=True), merge_mode='sum')(y_pred)
    #y_pred = Bidirectional(LSTM(NUM_HIDDEN), merge_mode='sum')(y_pred)
    model = Model(inputs=x,outputs=y_pred)
    model.compile(loss=ctc_loss, optimizer='adam', metrics=['acc'])
    model.summary()
    return model

def evaluate_model():
    pass

if __name__ == "__main__":
    if LOAD_PICKLE == False:
        # create the dataset
        MyTimitDataset = TIMITDataset()
        MyTimitDataset.load()
        MyTimitDataset.to_file()
    else:
        # just load it from pickle
        filename = glob.glob("timit*.pickle")[0]
        with open(filename, "rb") as dataset_file:
            timit_dataset = pickle.load(dataset_file)
        print(np.asarray(timit_dataset.X_train[0]).shape)
        print(timit_dataset.X_train[0])
        print(timit_dataset.y_train[0])

    #create_model_keras()
    evaluate_model()
