import tensorflow as tf
import math
from src.fc_layer import FCLayer
from src.nn_data import NNData
from src.atomic_nn import AtomicNN


class NN:
    def __init__(self, nn_config, train_config, info_config, datasets):
        self.nn_config = nn_config
        self.train_config = train_config
        self.datasets = datasets
        self.nn_data = NNData(datasets.data_config, datasets, info_config)
        self.train_op = None
        self.atomic_nns = []

        for atomic_nn_idx, atomic_nn_config in enumerate(self.nn_config):
            self.atomic_nns.append(AtomicNN(atomic_nn_config, train_config, datasets, atomic_nn_idx))

    def create_nn_graph(self, data_key):
        means = []
        vars = []
        for atomic_nn in self.atomic_nns:
            mean, var = atomic_nn.create_nn_graph(data_key)
            means.append(tf.expand_dims(mean, axis=1))
            if var.type != 'NoOp':
                vars.append(tf.expand_dims(vars, axis=1))

        # output statistics of atomic NNs are assumed to be independent
        means = tf.concat(means, axis=1)
        mean = tf.reduce_sum(means, axis=1)
        if len(vars) != 0:
            vars = tf.concat(vars, axis=1)
            var = tf.reduce_sum(vars, axis=1)

        # For the candidate dataset we are interested in mean and variance of output of NN
        if data_key == 'ca':
            if self.train_config['method']['name'] == 'pfp':
                return mean, var
            else:
                return mean, tf.no_op()

        t = self.datasets[data_key]['t']

        if self.train_config['type'] == 'pfp':
            y_m = tf.cast(mean, dtype=tf.float64)
            y_v = tf.cast(var, dtype=tf.float64)
            beta = self.train_config['beta']
            inv_beta = 1 / (2 * beta)

            # Expected log likelihood. The output for a given set of weights of the network
            # is assumed to be a Gaussian with fixed variance beta
            elogl = tf.reduce_mean(-tf.log(2 * math.pi * beta) -
                                   inv_beta * tf.square(t - y_m) -
                                   y_v * inv_beta)

            # The KL loss is scaled down (instead of scaling elogl up), such that resulting variational free energy
            # is invariant to the batch size. The hyperparameter 'data_m' is used for a tradeoff between KL and elogl
            kl = 0
            for layer in self.layers:
                kl = kl + layer.weights.get_kl_loss()
            kl /= (self.nn_config['data_m'] *
                   self.datasets.data_config['minibatch_size'] *
                   self.datasets.data_config['n_minibatches'])

            vfe = kl - elogl
            self.nn_data.add_metrics(data_key, vfe, kl, elogl)

        elif self.train_config['type'] == 'lr':
            y = tf.cast(mean, dtype=tf.float64)

            # The expected log likelihood is the L2 loss of residuals
            elogl = -tf.reduce_mean(tf.square(t - y))

            # The KL loss is scaled down (instead of scaling elogl up), such that resulting variational free energy
            # is invariant to the batch size. The hyperparameter 'data_m' is used for a tradeoff between KL and elogl
            kl = 0
            for layer in self.layers:
                kl = kl + layer.weights.get_kl_loss()
            kl /= (self.nn_config['data_m'] *
                   self.datasets.data_config['minibatch_size'] *
                   self.datasets.data_config['n_minibatches'])

            vfe = kl - elogl
            self.nn_data.add_metrics(data_key, vfe, kl, elogl)

        elif self.train_config['type'] == 'mcd':
            y = tf.cast(mean, dtype=tf.float64)

            # For simplicity we refer to the loss as vfe
            vfe = tf.reduce_mean(tf.square(t - y))
            self.nn_data.add_metrics(data_key, vfe, tf.no_op(), tf.no_op())

        return vfe

    # Creates network for training
    def create_training_graph(self, data_key):
        with tf.variable_scope(data_key):
            vfe = self.create_nn_graph(data_key)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.training_config['learning_rate'])
            gradients = optimizer.compute_gradients(vfe)
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










