import numpy as np
import pickle 
import data_utils

def collapse_activations(xs, ys, thresh=30):
    new_activations = []
    for x in xs:
        old_shape = x.shape
        x_shift = np.roll(x, -1)
        x_temp = x - x_shift
        x_temp = np.delete(x_temp, -1, axis=0) 

        mask = np.greater_equal(np.linalg.norm(x_temp, ord=2, axis=1), thresh)
        avg_activation = 0
        avg_i = 0
        out = []
        for i, xi in enumerate(x_temp):
            avg_activation += x[i]
            avg_i += 1
            if mask[i] == True:
                out.append(avg_activation / avg_i)
                avg_acc = 0
                avg_i = 0
        out = np.array(out)
        #print('Old shape: {} new shape: {}'.format(old_shape, out.shape))
        new_activations.append(out)
    np.array(new_activations)
    return new_activations
    


if __name__ == '__main__':
    xs, ys = data_utils.load_data('/home/cerrato/activations-allspeakers.pkl')
    new = collapse_activations(xs, ys)
    with open('activations-collapsed', 'wb') as f:
        pickle.dump(new, f)
