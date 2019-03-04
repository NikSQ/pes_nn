import tensorflow as tf
import numpy as np
import math
from src.fc_layer import FCLayer
from src.nn_data import NNData
from src.atomic_nn import AtomicNN


class NN:
    def __init__(self, nn_config, train_config, info_config, l_data):
        self.nn_config = nn_config
        self.train_config = train_config
        self.l_data = l_data
        self.nn_data = NNData(l_data, info_config)
        self.train_op = None
        self.atomic_nns = []
        self.gradient_summaries = None
        self.batch_size = tf.placeholder(name='batch_size', shape=(), dtype=tf.int32)

        for atomic_nn_idx, atom in enumerate(self.nn_config['atoms']):
            self.atomic_nns.append(AtomicNN(self.nn_config[atom], train_config, l_data, atomic_nn_idx, self.batch_size))
        # TODO: This needs to be changed if not all atomic NNs are of the same type
        self.sample_op = self.atomic_nns[0].sample_op
        self.create_nn_graph()

    def create_nn_graph(self):
        with tf.variable_scope('nn'):
            means = []
            vars = []
            for atomic_nn in self.atomic_nns:
                mean, var = atomic_nn.create_nn_graph()
                means.append(mean)
                vars.append(var)
            # output statistics of atomic NNs are assumed to be independent
            mean = tf.reduce_sum(tf.concat(means, axis=1), axis=1)
            var = tf.reduce_sum(tf.concat(vars, axis=1), axis=1)
            self.nn_data.add_output(mean, var)

            t = self.l_data.t

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

                kl = self.compute_kl()
                vfe = -elogl + kl
                self.nn_data.add_metrics(vfe, elogl_op=elogl, kl_op=kl)

            elif self.train_config['method'] == 'lr':
                y = tf.cast(mean, dtype=tf.float64)

                # The expected log likelihood is the negated L2 loss of residuals (likelihood is gaussian)
                elogl = -tf.reduce_mean(tf.square(t - y))

                kl = self.compute_kl()
                vfe = - elogl + kl
                self.nn_data.add_metrics(vfe, elogl_op=elogl, kl_op=kl)

            elif self.train_config['method'] == 'mcd':
                y = tf.cast(mean, dtype=tf.float64)

                # For the sake of reusing code we refer to the loss as vfe
                vfe = tf.reduce_mean(tf.square(t - y))
                self.sample_loss = t-y
                self.nn_data.add_metrics(vfe, 'vfe')

            optimizer = tf.train.AdamOptimizer(learning_rate=self.train_config['learning_rate'])
            trainable_variables = tf.trainable_variables()
            l2_loss = 0
            for var in trainable_variables:
                if var.name.endswith('w:0'):
                    l2_loss += tf.nn.l2_loss(var)

            gradients = optimizer.compute_gradients(vfe + self.nn_config['l2'] * tf.cast(l2_loss, dtype=tf.float64))

            summaries = []
            for grad, var in gradients:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.name, grad))
            self.gradient_summaries = tf.summary.merge(summaries)
            gradients = [(None if grad is None else tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gradients]
            self.train_op = optimizer.apply_gradients(gradients)

    # TODO: Compute KL loss by making a call to one atomic NN for each type of atom
    def compute_kl(self):
        # atoms, counts = np.unique(self.nn_config['atoms'], return_counts=True)
        kl = self.atomic_nns[0].compute_kl() * 55
        # kl = 0
        # for atomic_nn in self.atomic_nns:
        #     kl += atomic_nn.compute_kl()

        # The KL loss is scaled down (instead of scaling elogl up), such that resulting variational free energy
        # is invariant to the batch size. The hyperparameter 'data_m' is used for a tradeoff between KL and elogl
        kl /= (self.nn_config['data_m'] *
               self.l_data.data_config['tr']['minibatch_size'] *
               self.l_data.data_config['tr']['n_minibatches'])
        return tf.cast(kl, dtype=tf.float64)








