import tensorflow as tf
import math
from src.fc_layer import FCLayer
from src.nn_data import NNData


class AtomicNN:
    def __init__(self, atomic_nn_config, train_config, l_data, nn_idx, batch_size):
        self.atomic_nn_config = atomic_nn_config
        self.l_data = l_data
        self.train_config = train_config
        self.layers = []
        self.train_op = None
        self.nn_idx = nn_idx

        init_ops = []
        sample_ops = []
        with tf.variable_scope(self.atomic_nn_config['var_scope'], reuse=tf.AUTO_REUSE):
            for layer_idx, layer_config in enumerate(self.atomic_nn_config['layer_configs']):
                if layer_config['layer_type'] == 'fc':
                    layer = FCLayer(atomic_nn_config, layer_idx, batch_size)
                elif layer_config['layer_type'] == 'input':
                    continue
                else:
                    raise Exception('Layer type {} is not supported'.format(layer_config['layer_type']))
                init_ops.append(layer.init_ops)
                sample_ops.append(layer.sample_op)
                self.layers.append(layer)
        self.init_op = tf.group(init_ops)
        self.sample_op = tf.group(sample_ops)

    def create_nn_graph(self):
        # x_m and x_v are used to store input mean and variance to every layer
        # Note that only the pfp uses the variance of the input to a layer
        x_m = tf.squeeze(tf.slice(self.l_data.x, begin=(0, self.nn_idx, 0), size=(-1, 1, -1)))
        x_v = tf.fill(tf.shape(x_m), 0.)  # variance of input to network is by definition 0
        for layer_idx, layer in enumerate(self.layers, 1):
            if self.train_config['method'] == 'pfp':
                x_m, x_v = layer.create_pfp(x_m, x_v)
            elif self.train_config['method'] == 'lr':
                x_m = layer.create_lr_fp(x_m)
            elif self.train_config['method'] == 'mcd':
                x_m = layer.create_fp(x_m, self.l_data.is_training)

        return x_m, x_v

    def compute_kl(self):
        kl = 0
        for layer in self.layers:
            kl += layer.compute_kl(self.train_config['w_prior_v'])
        return kl

