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

        with tf.variable_scope(self.atomic_nn_config['var_scope'], reuse=True):
            for layer_idx, layer_config in enumerate(self.nn_config['layer_configs'], 1):
                if layer_config['layer_type'] == 'fc':
                    layer = FCLayer(atomic_nn_config, layer_idx)
                else:
                    raise Exception('Layer type {} is not supported'.format(layer_config['layer_type']))
                self.layers.append(layer)

    def create_nn_graph(self, data_key):
        # x_m and x_v will be used to store input mean and variance to every layer
        # Note that only the pfp uses the variance of the input to a layer
        x_m = tf.slice(self.datasets[data_key]['x'], begin=(0, self.nn_idx, 0), size=(-1, 1, -1))
        x_v = tf.fill(tf.shape(x_m), 0.)  # variance of input to network is by definition 0

        for layer_idx, layer in enumerate(self.layers, 1):
            if self.train_config['method']['name'] == 'pfp':
                x_m, x_v = layer.create_pfp(x_m, x_v)
            elif self.train_config['method']['name'] == 'lr':
                x_m = layer.create_lr_fp(x_m)
            elif self.train_config['method']['name'] == 'mcd':
                x_m = layer.create_fp(x_m)
        return x_m, x_v

