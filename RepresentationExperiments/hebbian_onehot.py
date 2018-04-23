from models.som.SOM import SOM
from models.som.HebbianModel import HebbianModel
from utils.constants import Constants
from utils.utils import from_csv_with_filenames, from_csv_visual, from_csv, to_csv
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from data.dataset import OneHotDataset

import os
import numpy as np

soma_path = os.path.join(Constants.DATA_FOLDER, 'onehot', 'audio_model', '')
somv_path = os.path.join(Constants.DATA_FOLDER, 'onehot', 'visual_model', '')
hebbian_path = os.path.join(Constants.DATA_FOLDER, 'onehot', 'hebbian_model', '')

if __name__ == '__main__':
    acc = []
    for i in range(10):
        n_classes = 4
        dataset = OneHotDataset(n_classes)
        a_xs = dataset.x
        a_ys = dataset.y
        v_xs = dataset.x
        v_ys = dataset.y
        # scale audio data to 0-1 range
        a_xs = MinMaxScaler().fit_transform(a_xs)
        v_xs = MinMaxScaler().fit_transform(v_xs)
        a_dim = len(a_xs[0])
        v_dim = len(v_xs[0])
        som_a = SOM(5, 5, a_dim, checkpoint_dir=soma_path, n_iterations=100)
        som_v = SOM(5, 5, v_dim, checkpoint_dir=somv_path, n_iterations=100)
        som_a.train(a_xs)
        som_v.train(v_xs)
        hebbian_model = HebbianModel(som_a, som_v, a_dim=a_dim,
                                     v_dim=v_dim, n_presentations=1, learning_rate=1, n_classes=n_classes,
                                     checkpoint_dir=hebbian_path)
        print('Training...')
        hebbian_model.train(a_xs, v_xs)
        print('Evaluating...')
        accuracy = hebbian_model.evaluate(a_xs, v_xs, a_ys, v_ys, source='a', img_path = './')
        acc.append(accuracy)
        print('n={}, accuracy={}'.format(1, accuracy))
    print(sum(acc)/len(acc))
