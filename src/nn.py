import tensorflow as tf
import numpy as np
import math
from src.fc_layer import FCLayer
from src.nn_data import NNData
from src.atomic_nn import AtomicNN


class NN:
    def __init__(self, nn_config, train_config, info_config, datasets):
        self.nn_config = nn_config
        self.train_config = train_config
        self.datasets = datasets
        self.nn_data = NNData(datasets, info_config)
        self.train_op = None
        self.atomic_nns = []
        self.gradient_summaries = None

        for atomic_nn_idx, atom in enumerate(self.nn_config['atoms']):
            self.atomic_nns.append(AtomicNN(self.nn_config[atom], train_config, datasets, atomic_nn_idx))

        for data_key in datasets.sets.keys():
            if 'tr' in data_key:
                self.create_training_graph(data_key)
            else:
                self.create_evaluation_graph(data_key)

    def create_nn_graph(self, data_key):
        means = []
        vars = []
        for atomic_nn in self.atomic_nns:
            mean, var = atomic_nn.create_nn_graph(data_key)
            means.append(mean)
            vars.append(var)
        # output statistics of atomic NNs are assumed to be independent
        mean = tf.reduce_sum(tf.concat(means, axis=1), axis=1)
        var = tf.reduce_sum(tf.concat(vars, axis=1), axis=1)
        # For the candidate dataset we are interested in mean and variance of output of NN
        if data_key == 'ca':
            if self.train_config['method'] == 'pfp':
                return mean, var
            else:
                return mean, tf.no_op() # Note that this is not the mean, just an output after weights were sampled

        t = self.datasets.sets[data_key]['t']

        if self.train_config['method'] == 'pfp':
            y_m = tf.cast(mean, dtype=tf.float64)
            y_v = tf.cast(var, dtype=tf.float64)
            beta = np.float64(self.train_config['beta'])
            inv_beta = 1 / (2 * beta)

            # Expected log likelihood. The output for a given set of weights of the network
            # is assumed to be a Gaussian with fixed variance beta
            elogl = tf.reduce_mean(-tf.log(2 * math.pi * beta) -
                                   inv_beta * tf.square(t - y_m) -
                                   y_v * inv_beta)

            kl = self.compute_kl(data_key)
            vfe = -elogl
            self.nn_data.add_metrics(data_key, vfe, elogl_op=elogl)

        elif self.train_config['method'] == 'lr':
            y = tf.cast(mean, dtype=tf.float64)

            # The expected log likelihood is the negated L2 loss of residuals (likelihood is gaussian)
            elogl = -tf.reduce_mean(tf.square(t - y))

            kl = self.compute_kl(data_key)
            vfe = - elogl
            self.nn_data.add_metrics(data_key, vfe, elogl_op=elogl)

        elif self.train_config['method'] == 'mcd':
            y = tf.cast(mean, dtype=tf.float64)

            # For the sake of reusing code we refer to the loss as vfe
            vfe = tf.reduce_mean(tf.square(t - y))
            self.nn_data.add_metrics(data_key, vfe)

        return vfe

    # Creates network for training
    def create_training_graph(self, data_key):
        with tf.variable_scope(data_key):
            vfe = self.create_nn_graph(data_key)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.train_config['learning_rate'])
            gradients = optimizer.compute_gradients(vfe)

            summaries = []
            for grad, var in gradients:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.name, grad))
            self.gradient_summaries = tf.summary.merge(summaries)
            self.train_op = optimizer.apply_gradients(gradients)

    # Creates a network for evaluation only (no backward pass)
    # The performance (loss / variational free energy) is recorded for validation and test set
    # The output (mean and variance of potential) is recorded for candidate set (data_key = 'ca')
    def create_evaluation_graph(self, data_key):
        with tf.variable_scope(data_key):
            if data_key != 'ca':
                self.create_nn_graph(data_key)
            else:
                y_m, y_v = self.create_nn_graph(data_key)
                self.nn_data.add_output(data_key, y_m, y_v)

    # TODO: Compute KL loss by making a call to one atomic NN for each type of atom
    def compute_kl(self, data_key):
        atoms, counts = np.unique(self.nn_config['atoms'], return_counts=True)
        kl = 0
        #for atom, count in zip(atoms, counts):

        # The KL loss is scaled down (instead of scaling elogl up), such that resulting variational free energy
        # is invariant to the batch size. The hyperparameter 'data_m' is used for a tradeoff between KL and elogl

        kl /= (self.nn_config['data_m'] *
            self.datasets.sets[data_key]['minibatch_size'] *
            self.datasets.sets[data_key]['n_minibatches'])
        #kl = tf.zeros((1), dtype=tf.float64) # line is currently needed
        return kl







