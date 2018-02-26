import tensorflow as tf
import numpy as np
import os

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

            delta = 1 - tf.exp(-self.learning_rate * tf.transpose(self.activation_a) * self.activation_v)
            new_weights = tf.add(self.weights, delta)
            self.training = tf.assign(self.weights, new_weights)

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
            for i in range(self.n_presentations):
                print('Presentation {}'.format(i+1))
                activation_a, _ = self.som_a.get_activations(input_a[i])
                activation_v, _ = self.som_v.get_activations(input_v[i])
                self._sess.run(self.training,
                               feed_dict={self.activation_a: activation_a,
                                          self.activation_v: activation_v})
            self._trained = True

            if self.checkpoint_dir != None:
                saver = tf.train.Saver()
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                saver.save(self._sess,
                           os.path.join(self.checkpoint_dir,
                                       'model.ckpt'),
                           1)

    def restore_trained(self):
        pass

    def evaluate(self, input_a, input_v):
        pass
