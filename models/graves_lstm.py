import tensorflow as tf
import librosa as lr
import pickle
import glob
import time
import numpy as np
from data.dataset import TIMITDataset
from keras.layers import Input, Bidirectional, LSTM
from keras.models import Sequential, Model
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn, LSTMCell


LOAD_PICKLE = False
NUM_LAYERS = 2
NUM_HIDDEN = 50
BATCH_SIZE = 32
NUM_EPOCHS = 100

def inference():
    pass

def loss():
    pass

def training():
    pass
# TODO:
# https://stackoverflow.com/questions/34156639/tensorflow-python-valueerror-setting-an-array-element-with-a-sequence-in-t
def create_model(dataset):
    max_time_length = max([t for (t, f) in [x.shape for x in dataset.X_train]])
    num_features = dataset.X_train[0].shape[1]
    num_classes = max(TIMITDataset.phoneme_dict.values())
    num_examples = len(dataset.X_train)
    
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=(BATCH_SIZE, max_time_length, num_features))
        targets = tf.sparse_placeholder(tf.int32)
        seq_length = tf.placeholder(tf.int32)

        lstm_cell_forward_list = []
        lstm_cell_backward_list = []
        for i in range(0, NUM_LAYERS):
            lstm_cell_forward_list.append(LSTMCell(NUM_HIDDEN))
            lstm_cell_backward_list.append(LSTMCell(NUM_HIDDEN))

        outputs, f_state, b_state = stack_bidirectional_dynamic_rnn(lstm_cell_forward_list, lstm_cell_backward_list,
                                        inputs, dtype=tf.float32)
        
        # prepare the last fully-connected layer, which weights are shared throughout the time steps
        outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])
        W = tf.Variable(tf.truncated_normal([NUM_HIDDEN,
                                             num_classes],
                                            stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        fc_out = tf.matmul(outputs, W) + b
        fc_out = tf.reshape(fc_out, [BATCH_SIZE, -1, num_classes]) # Reshaping back to the original shape
        
        loss = tf.nn.ctc_loss(targets, fc_out, seq_length)
        cost = tf.reduce_mean(loss)

        optimizer = tf.train.MomentumOptimizer(0.05,
                                               0.9).minimize(cost)

        # Option 2: tf.nn.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(fc_out, seq_length)
        
        # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              targets))
        
    with tf.Session(graph=graph) as session:
        # Initializate the weights and biases
        tf.global_variables_initializer().run()

        for curr_epoch in range(NUM_EPOCHS):
            train_cost = train_ler = 0
            start = time.time()

            num_batches_per_epoch = int(num_examples/BATCH_SIZE)
            for batch in range(num_batches_per_epoch):
                # prepare data and targets
                start_index = batch * BATCH_SIZE
                end_index = (batch + 1) * BATCH_SIZE 

                if end_index >= len(dataset.X_train) - 1:
                    end_index = len(dataset.X_train) - 1
                
                batch_seq_length = [x.shape[0] for x in dataset.X_train[start_index:end_index]]

                feed = {inputs: dataset.X_train[start_index:end_index],
                        targets: tf.one_hot(dataset.y_train[start_index:end_index], num_classes),
                        seq_length: batch_seq_length}

                batch_cost, _ = session.run([cost, optimizer], feed)
                train_cost += batch_cost*batch_size
                train_ler += session.run(ler, feed_dict=feed)*BATCH_SIZE

            train_cost /= num_examples
            train_ler /= num_examples

            log = "Epoch "+str(curr_epoch)+", train_cost = {:.3f}, train_ler = {:.3f} time = {:.3f}"
            print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                             time.time() - start))

    return 

def create_model_keras():
    x = Input(shape=(NUM_FEATURES, None, None))
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
        #MyTimitDataset.to_file()
    else:
        # just load it from pickle
        filename = glob.glob("timit*.pickle")[0]
        with open(filename, "rb") as dataset_file:
            MyTimitDataset = pickle.load(dataset_file)
    #print(timit_dataset.X_train)
    #create_model(MyTimitDataset)
    evaluate_model()
