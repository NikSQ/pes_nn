import tensorflow as tf
import math
from src.fc_layer import FCLayer
from src.nn_data import NNData


class AtomicNN:
    def __init__(self, atomic_nn_config, train_config, datasets, nn_idx):
        self.atomic_nn_config = atomic_nn_config
        self.datasets = datasets
        self.train_config = train_config
        self.layers = []
        self.train_op = None
        self.nn_idx = nn_idx

        with tf.variable_scope(self.atomic_nn_config['var_scope'], reuse=tf.AUTO_REUSE):
            for layer_idx, layer_config in enumerate(self.atomic_nn_config['layer_configs']):
                if layer_config['layer_type'] == 'fc':
                    layer = FCLayer(atomic_nn_config, layer_idx)
                elif layer_config['layer_type'] == 'input':
                    continue
                else:
                    raise Exception('Layer type {} is not supported'.format(layer_config['layer_type']))
                self.layers.append(layer)

    def create_nn_graph(self, data_key):
        with tf.name_scope(self.atomic_nn_config['var_scope']):
            # x_m and x_v are used to store input mean and variance to every layer
            # Note that only the pfp uses the variance of the input to a layer
            x_m = tf.squeeze(tf.slice(self.datasets.sets[data_key]['x'], begin=(0, self.nn_idx, 0), size=(-1, 1, -1)))
            self.x_m = x_m
            x_v = tf.fill(tf.shape(x_m), 0.)  # variance of input to network is by definition 0
            for layer_idx, layer in enumerate(self.layers, 1):
                if self.train_config['method'] == 'pfp':
                    x_m, x_v = layer.create_pfp(x_m, x_v)
                elif self.train_config['method'] == 'lr':
                    x_m = layer.create_lr_fp(x_m)
                elif self.train_config['method'] == 'mcd':
                    x_m = layer.create_fp(x_m)

            return x_m, x_v

    # TODO: Implement KL loss
    def compute_kl(self):
        return 0

