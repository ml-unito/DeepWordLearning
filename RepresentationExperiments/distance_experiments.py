import numpy as np
import os
from utils.utils import from_csv_with_filenames
from utils.constants import Constants
import matplotlib.pyplot as plt
from collections import OrderedDict

CSV_PATH = os.path.join(
            Constants.DATA_FOLDER,
            '10classes',
            'audio_data.csv'
            )

def get_prototypes(xs, ys):
    prototype_dict = {unique_y: [] for unique_y in set(ys)}
    for i, x in enumerate(xs):
        prototype_dict[ys[i]].append(x)
    prototype_dict = {k: np.array(prototype) for k, prototype in prototype_dict.items()}
    result = OrderedDict(sorted(prototype_dict.items()))
    for y in set(sorted(ys)):
        result[y] = np.mean(prototype_dict[y], axis=0)
    return result

def average_prototype_distance_matrix(xs, ys, filenames):
    """
        Relies on all y in ys being in the interval [0, number of classes)
        As a general rule, x in xs and y in ys are not ordered by class but by speaker
        That is why I am not using models.som.SOMTest.classPrototype, which
        has the opposite assumption
    """
    prototype_distance_matrix = np.zeros((len(set(ys)), len(set(ys))))
    # compute prototypes in dictionary d
    prototype_dict = {unique_y: [] for unique_y in set(ys)}
    for i, x in enumerate(xs):
        prototype_dict[ys[i]].append(x)
    prototype_dict = {k: np.array(prototype) for k, prototype in prototype_dict.items()}
    for y in set(ys):
        prototype_dict[y] = np.mean(prototype_dict[y], axis=0)
    prototypes = np.asarray(list(prototype_dict.values())).T
    for i, x in enumerate(xs):
        prototype_distance_matrix[ys[i]][:] += np.mean(np.absolute(prototypes - x.reshape((-1, 1))), axis=0).T
    print(prototype_distance_matrix)
    plt.matshow(prototype_distance_matrix, cmap=plt.get_cmap('Greys'))
    plt.show()

def examples_distance(xs, i1, i2):
    return np.linalg.norm(xs[i1]-xs[i2])

if __name__ == '__main__':
    xs, ys, filenames = from_csv_with_filenames(CSV_PATH)
    ys = [int(y)-1000 for y in ys] # see comment above

    average_prototype_distance_matrix(xs, ys, filenames)
    #i1 = 23
    #i2 = 163
    #d = examples_distance(xs, i1, i2)
    #print(d)
