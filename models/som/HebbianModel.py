import tensorflow as tf
import numpy as np
import os
import sys

class HebbianModel(object):

    def __init__(self, som_a, som_v, a_dim, v_dim, learning_rate=10,
                 n_presentations=1,
                 checkpoint_dir=None):
        assert som_a._m == som_v._m and som_a._n == som_v._n
        self.num_neurons = som_a._m * som_a._n
        self._graph = tf.Graph()
        self.som_a = som_a
        self.som_v = som_v
        self.a_dim = a_dim
        self.v_dim = v_dim
        self.n_presentations = n_presentations
        self.checkpoint_dir = checkpoint_dir
        self.learning_rate = learning_rate
        self._trained = False

        with self._graph.as_default():
            self.weights = tf.Variable(
                             tf.random_normal([self.num_neurons, self.num_neurons],
                             mean=1/self.num_neurons,
                             stddev=1/np.sqrt(1000*self.num_neurons))
                           )

            self.activation_a = tf.placeholder(dtype=tf.float32, shape=[self.num_neurons])
            self.activation_v = tf.placeholder(dtype=tf.float32, shape=[self.num_neurons])
            self.assigned_weights = tf.placeholder(dtype=tf.float32, shape=[self.num_neurons, self.num_neurons])

            self.delta = 1 - tf.exp(-self.learning_rate * tf.matmul(tf.reshape(self.activation_a, (-1, 1)), tf.reshape(self.activation_v, (1, -1))))
            new_weights = tf.add(self.weights, self.delta)
            self.training = tf.assign(self.weights, new_weights)

            self.assign_op = tf.assign(self.weights, self.assigned_weights)

            self._sess  = tf.Session()
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def train(self, input_a, input_v):
        '''
        input_a: list containing a number of training examples equal to
                 self.n_presentations
        input_v : same as above
        '''
        assert len(input_a) == len(input_v) == self.n_presentations, \
               'Number of training examples and number of desired presentations \
                is incoherent. len(input_a) = {}; len(input_v) = {}; \
                n_presentations = {}'.format(len(input_a), len(input_v),
                                             self.n_presentations)
        with self._sess:
            # present images to model
            for i in range(self.n_presentations):
                print('Presentation {}'.format(i+1))
                activation_a, _ = self.som_a.get_activations(input_a[i])
                activation_v, _ = self.som_v.get_activations(input_v[i])
                _, d = self._sess.run([self.training, self.delta],
                               feed_dict={self.activation_a: activation_a,
                                          self.activation_v: activation_v})

            # normalize sum of weights to 1
            w = self._sess.run(self.weights)
            w = w.flatten()
            w_sum = np.sum(w)
            w_norm = [wi / w_sum for wi in w]
            w = np.reshape(w_norm, (self.num_neurons, self.num_neurons))

            self._sess.run(self.assign_op, feed_dict={self.assigned_weights: w})

            self._trained = True

            # save to checkpoint_dir
            if self.checkpoint_dir != None:
                saver = tf.train.Saver()
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                saver.save(self._sess,
                           os.path.join(self.checkpoint_dir,
                                       'model.ckpt'),
                           1)

            # convert weights to numpy arrays from tf tensors
            self.weights = self._sess.run(self.weights)

    def get_bmu_propagate(self, x, source_som='v'):
        '''
        Get the best matching unit by propagating an input vector's activations
        to the other SOM. More specifically, we use the synapses connected to the
        source som's BMU to find a matching BMU in the target som.

        x: input vector. Must have a compatible size with the som described in
           'source_som' parameter
        source_som: a string representing the source som. If 'a', the activations
        of the audio som will be propagated to the visual one; if 'v', the opposite
        will happen.
        '''
        if source_som == 'v':
            from_som = self.som_v
            to_som = self.som_a
        elif source_som == 'a':
            from_som = self.som_a
            to_som = self.som_v
        else:
            raise ValueError('Wrong string for source_som parameter')
        source_activation, _ = from_som.get_activations(x)
        bmu_index = np.argmax(np.array(source_activation))
        #bmu_weights = self._sess.run(source_som._weightage_vects[bmu_index]) # probably un-needed?
        if source_som == 'v':
            hebbian_weights = self.weights[:][bmu_index]
        else:
            hebbian_weights = self.weights[bmu_index][:]
        target_activation = hebbian_weights * source_activation
        try:
            assert target_activation.shape[0] == (to_som._n * to_som._m)
        except AssertionError:
            print('Shapes do not match. target_activation: {};\
                   som: {}'.format(target_activation.shape, to_som._n * to_som._m))
            sys.exit(1)
        return np.argmax(target_activation)

    def restore_trained(self):
        pass

    def evaluate(self, input_a, input_v):
        pass

# some test cases. do not use as an entry point for experiments!
if __name__ == '__main__':
    pass
