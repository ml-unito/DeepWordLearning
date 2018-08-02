# Copyright 2017 Giorgia Fenoglio, Mattia Cerrato
#
# This file is part of NNsTaxonomicResponding.
#
# NNsTaxonomicResponding is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NNsTaxonomicResponding is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NNsTaxonomicResponding.  If not, see <http://www.gnu.org/licenses/>.

import tensorflow as tf
import numpy as np
import math
import os
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 8})
import matplotlib.pyplot as plt
from utils.constants import Constants
from matplotlib import colors


class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """

    #To check if the SOM has been trained
    _trained = False


    def __init__(self, m, n, dim, checkpoint_dir=None, n_iterations=50, alpha=None, sigma=None,
                 tau=0.5, threshold=0.6, batch_size=500):
        """
        Initializes all necessary components of the TensorFlow
        Graph.

        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """

        #Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)

        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)

        self.tau = tau
        self.threshold = threshold

        self.batch_size = batch_size

        self._n_iterations = abs(int(n_iterations))

        if checkpoint_dir is None:
          self.checkpoint_dir = './model100ClassesVisivo/'
        else:
          self.checkpoint_dir = checkpoint_dir

        ##INITIALIZE GRAPH
        self._graph = tf.Graph()

        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE

            #Randomly initialized weightage vectors for all neurons,
            #stored together as a matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal(
                [m*n, dim]))

            #Matrix of size [m*n, 2] for SOM grid locations
            #of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))

            ##PLACEHOLDERS FOR TRAINING INPUTS
            #We need to assign them as attributes to self, since they
            #will be fed in during training

            #The training vectors
            self._vect_input = tf.placeholder("float", [None, dim])
            #Iteration number
            self._iter_input = tf.placeholder("float")

            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            #Only the final, 'root' training op needs to be assigned as
            #an attribute to self, since all the rest will be executed
            #automatically during training

            bmu_indexes = self._get_bmu()

            #This will extract the location of the BMU based on the BMU's
            #index. This has dimensionality [batch_size, 2] where 2 is (i, j),
            #the location of the BMU in the map
            bmu_loc = tf.gather(self._location_vects, bmu_indexes)

            #To compute the alpha and sigma values based on iteration
            #number
            learning_rate = 1.0 - tf.div(self._iter_input, self._n_iterations)
            _alpha_op = alpha * learning_rate
            _sigma_op = (sigma * learning_rate) ** 2

            #Construct the op that will generate a vector with learning
            #rates for all neurons, based on iteration number and location
            #wrt BMU.

            #Tensor of shape [batch_size, num_neurons] containing the distances
            #between the BMU and all other neurons, for each batch
            bmu_distance_squares = self._get_bmu_distances(bmu_loc)

            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), _sigma_op)))
            learning_rate_op = _alpha_op * neighbourhood_func

            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input
            learning_rate_matrix = _alpha_op * neighborhood_func

            weightage_delta = self._get_weight_delta(learning_rate_matrix)

            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)

            ##INITIALIZE SESSION
            config = tf.ConfigProto(
                  device_count = {'GPU': 0}
              )
            self._sess  = tf.Session(config=config)


            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def _get_weight_delta(self, learning_rate_marix):
        """
        """
        diff_matrix = tf.cast(self.weightage_vects - tf.expand_dims(self._vect_input, 1), "float32")
        delta = tf.reduce_mean(tf.expand_dims(learning_rate_matrix, 2) * diff_matrix, 0)
        return delta

    def _get_bmu_distances(self, bmu_loc):
        """
        """
        squared_distances = tf.reduce_sum((_location_vects - tf.expand_dims(bmu_loc, 1)) ** 2, 2)
        return squared_distances

    def _get_bmu(self):
        """
        Returns the BMU for each example in self._vect_input. The return value's dimensionality
        is therefore [batch_size]
        """
        squared_differences = (self._weightage_vects - tf.expand_dims(self._vect_input, 1)) ** 2
        squared_distances = tf.reduce_sum(squared_differences, 2)
        bmu_index = tf.argmin(squared_distances, 1)
        return bmu_index

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        #Nested iterations over both dimensions
        #to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
        with self._sess:
          #Training iterations
          for iter_no in range(self._n_iterations):
              if iter_no % 10 == 0:
                  print('Iteration {}'.format(iter_no))
              #Train with each vector one by one
              count = 0
              for input_vect in input_vects:
                  count = count + 1
                  self._sess.run(self._training_op,
                                 feed_dict={self._vect_input: input_vect,
                                            self._iter_input: iter_no})
          #Store a centroid grid for easy retrieval later on
          centroid_grid = [[] for i in range(self._m)]
          self._weightages = list(self._sess.run(self._weightage_vects))
          self._locations = list(self._sess.run(self._location_vects))
          for i, loc in enumerate(self._locations):
              centroid_grid[loc[0]].append(self._weightages[i])
          self._centroid_grid = centroid_grid

          self._trained = True

          # Store the trained model
          saver = tf.train.Saver()
          if not os.path.exists(self.checkpoint_dir):
              os.makedirs(self.checkpoint_dir)
          saver.save(self._sess,
                     os.path.join(self.checkpoint_dir,
                                 'model.ckpt'),
                     1)


    def restore_trained(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            with self._sess:
              saver = tf.train.Saver()
              saver.restore(self._sess, ckpt.model_checkpoint_path)

              #restore usefull variable
              centroid_grid = [[] for i in range(self._m)]
              self._weightages = list(self._sess.run(self._weightage_vects))
              self._locations = list(self._sess.run(self._location_vects))
              for i, loc in enumerate(self._locations):
                  centroid_grid[loc[0]].append(self._weightages[i])
              self._centroid_grid = centroid_grid

              self._trained = True

              print('RESTORED SOM MODEL')
              return True
        else:
            print('NO CHECKPOINT FOUND')
            return False


    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        if not self._trained:
            raise ValueError("SOM not trained yet")

        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect-
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])

        return to_return


    def get_BMU(self, input_vect):
        min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(input_vect-
                                                         self._weightages[x]))

        return [min_index,self._locations[min_index]]

    def detect_superpositions(self, l):
        for l_i in l:
            if len(l_i) > 1:
                if all(x == l_i[0] for x in l_i) == False:
                    return True
        return False

    def memorize_examples_by_class(self, X, y):
        self.bmu_class_dict = {i : [] for i in range(self._n * self._m)}
        for i, (x, yi) in enumerate(zip(X, y)):
            activations, _ = self.get_activations(x, normalize=False, mode='exp', threshold=False)
            bmu_index = np.argmax(activations)
            self.bmu_class_dict[bmu_index].append(yi)
        superpositions = self.detect_superpositions(self.bmu_class_dict.values())
        print('More than a class mapped to a neuron: '+ str(superpositions))
        return superpositions

    def get_activations(self, input_vect, normalize=True, threshold=True, mode='exp'):
      # get activations for the word learning

  # Quantization error:
      activations = list()
      pos_activations = list()
      for i in range(len(self._weightages)):
          d = np.array([])

          d = (np.absolute(input_vect-self._weightages[i])).tolist()
          if mode == 'exp':
              activations.append(math.exp(-(np.sum(d)/len(d))/self.tau))
          if mode == 'linear':
              activations.append(1/np.sum(d))
          pos_activations.append(self._locations[i])
      activations = np.array(activations)
      if normalize:
          max_ = max(activations)
          min_ = min(activations)
          activations = (activations - min_) / float(max_ - min_)
      if threshold:
          idx = activations < self.threshold
          activations[idx] = 0
      return [activations,pos_activations]



    def plot_som(self, X, y, plot_name='som-viz.png'):
        image_grid = np.zeros(shape=(self._n,self._m))

        color_names = \
            {0: 'black', 1: 'blue', 2: 'skyblue',
             3: 'aqua', 4: 'darkgray', 5: 'green', 6: 'red',
             7: 'cyan', 8: 'violet', 9: 'yellow'}
        #Map colours to their closest neurons
        mapped = self.map_vects(X)

        #Plot
        plt.imshow(image_grid)
        plt.title('Color SOM')
        for i, m in enumerate(mapped):
            plt.text(m[1], m[0], color_names[y[i]], ha='center', va='center',
                     bbox=dict(facecolor=color_names[y[i]], alpha=0.5, lw=0))
        plt.savefig(os.path.join(Constants.PLOT_FOLDER, plot_name))
