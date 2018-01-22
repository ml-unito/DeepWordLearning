import tensorflow as tf
#from deepspeech.model import Model
import scipy.io.wavfile as wav
import sys
import glob
import argparse
import json
import re
import glob
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from deepspeech.utils import audioToInputVector
import scipy.io.wavfile as wav
import pickle

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


def create_model():
    # These constants control the beam search decoder

    # Beam width used in the CTC decoder when building candidate transcriptions
    BEAM_WIDTH = 500

    # The alpha hyperparameter of the CTC decoder. Language Model weight
    LM_WEIGHT = 1.75

    # The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
    WORD_COUNT_WEIGHT = 1.00

    # Valid word insertion weight. This is used to lessen the word insertion penalty
    # when the inserted word is part of the vocabulary
    VALID_WORD_COUNT_WEIGHT = 1.00


    # These constants are tied to the shape of the graph used (changing them changes
    # the geometry of the first layer), so make sure you use the same constants that
    # were used during training

    deepspeech_model = Model(args.model, N_FEATURES, N_CONTEXT, args.alphabet, BEAM_WIDTH)
    
    if args.lm and args.trie:
        deepspeech_model.enableDecoderWithLM(args.alphabet, args.lm, args.trie, LM_WEIGHT,
                               WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)
    
    return deepspeech_model


def test_model(deepspeech_model):
    # load transcription data
    transcriptions = json.load(open(args.transcription_json))
    
    i = 0
    for i in range(len(transcriptions)):
        audio_file = args.audio_folder + '/' + str(i) + '.wav'
        sr, audio = wav.read(audio_file)

        assert sr == 16000, "Sample rate is not 16k! All other sample rates are unsupported as of now."

        audio_length = len(audio) * (1 / sr)
        
        y = deepspeech_model.stt(audio, sr)
        y_true = transcriptions[str(i)].split(',')[0]

        print('Estimated: {} \n True: {}'.format(y, y_true))

        i += 1

def get_model_output(filename):
    with tf.gfile.GFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    with tf.Graph().as_default() as graph:
        tensors_and_ops = tf.import_graph_def(graph_def, name='')
        with tf.Session(graph=graph) as sess:
            output_op = graph.get_operation_by_name('Minimum_3').outputs[0]
            input_op = graph.get_operation_by_name('input_node')
            input_length_op = graph.get_operation_by_name('input_lengths')

            batch_size = 1
            dropout = [0.] * 6
            #d = TIMITDataset()
            #d.load(n_mfcc=13)
            activations = {}
            for idx, filename in enumerate(glob.glob('/home/cerrato/DeepWordLearning/data/audio/*wav*/*.wav')):
                fs, audio = wav.read(filename)
                x = audioToInputVector(audio, fs, N_FEATURES, N_CONTEXT)
                out = sess.run(output_op, {'input_node:0': [x], 'input_lengths:0': [len(x)]})
                name = filename.split('/')[-2:]
                name = '/'.join(name)
                name = name.replace('.wav', '')
                activations[name] = out
                if idx % 50 == 0:
                    print(name)
            pickle.dump(activations, open('activations.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            act_load = pickle.load(open('activations.pkl', 'rb'))
    
    #with tf.Session() as sess:
    #    sess.run(tf.global_variables_initializer())
    #    with gfile.FastGFile(filename, 'rb') as f:
    #        graph_def = tf.GraphDef()
    #        graph_def.ParseFromString(f.read())
    #        g_in = tf.import_graph_def(graph_def)
    #        
    #        print([op.name for op in g_in.get_operations() if op.op_def and op.op_def.name=='Variable'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test script running the deepspeech model on the OS X speaker dataset.')
    parser.add_argument('model', type=str,
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('audio_folder', type=str,
                        help='Path to the audio directory containing the files (WAV format)')
    parser.add_argument('transcription_json', type=str,
                        help='Path to the json file containing the transcriptions of the WAV files')
    parser.add_argument('alphabet', type=str,
                        help='Path to the configuration file specifying the alphabet used by the network')
    parser.add_argument('lm', type=str, nargs='?',
                        help='Path to the language model binary file')
    parser.add_argument('trie', type=str, nargs='?',
                        help='Path to the language model trie file created with native_client/generate_trie')
    args = parser.parse_args()
    
    #deepspeech_model = create_model()
    #test_model(deepspeech_model)
    get_model_output(args.model)
