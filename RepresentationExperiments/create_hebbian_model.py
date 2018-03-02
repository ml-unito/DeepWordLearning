from models.som.SOM import SOM
from models.som.HebbianModel import HebbianModel
from utils.constants import Constants
from utils.utils import from_csv_with_filenames, from_csv_visual, from_csv
from sklearn.utils import shuffle
import os

soma_path = os.path.join(Constants.DATA_FOLDER, '10classes', 'audio_model', '')
somv_path = os.path.join(Constants.DATA_FOLDER, '10classes', 'visual_model', '')
hebbian_path = os.path.join(Constants.DATA_FOLDER, '10classes', 'hebbian_model', '')
audio_data_path = os.path.join(Constants.DATA_FOLDER,
                               '10classes',
                               'audio_prototypes.csv')
visual_data_path = os.path.join(Constants.DATA_FOLDER,
                                '10classes',
                                'VisualInputTrainingSet.csv')

if __name__ == '__main__':
    a_xs, a_ys = from_csv(audio_data_path)
    v_xs, v_ys = from_csv_visual(visual_data_path)
    a_xs, a_ys = shuffle(a_xs, a_ys, random_state=26)
    v_xs, v_ys = shuffle(v_xs, v_ys, random_state=26)
    a_dim = len(a_xs[0])
    v_dim = len(v_xs[0])
    som_a = SOM(20, 30, a_dim, checkpoint_dir=soma_path)
    som_v = SOM(20, 30, v_dim, checkpoint_dir=somv_path)
    som_a.restore_trained()
    som_v.restore_trained()
    hebbian_model = HebbianModel(som_a, som_v, a_dim=a_dim,
                                 v_dim=v_dim, n_presentations=10,
                                 checkpoint_dir=hebbian_path)
    hebbian_model.train(a_xs, v_xs)
    hebbian_model.evaluate(a_xs, v_xs, a_ys, v_ys, source='v')
