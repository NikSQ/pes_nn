import tensorflow as tf
from src.pfp_tools import approx_activation, transform_sig_activation, transform_tanh_activation
import numpy as np

# TODO: Add support for ReLU


def get_initializer(init_scheme, shape):
    if init_scheme == 'xavier':
        init_vals = np.random.randn(shape[0], shape[1]) * np.sqrt(2/sum(shape))
    elif init_scheme == 'zeros':
        init_vals = np.zeros(shape)
    elif type(init_scheme) is float:
        init_vals = np.ones(shape) * init_scheme
    else:
        raise Exception('{} is not a valid initialization scheme'.format(init_scheme))
    return tf.constant_initializer(init_vals)


class FCLayer:
    def __init__(self, atomic_nn_config, layer_idx):
        self.layer_config = atomic_nn_config['layer_configs'][layer_idx]
        self.w_shape = (atomic_nn_config['layout'][layer_idx-1], atomic_nn_config['layout'][layer_idx])
        self.b_shape = (1, self.w_shape[1])

        with tf.variable_scope(self.layer_config['var_scope']):
            # Normal distribution used for reparametrization
            self.gauss = tf.distributions.Normal(loc=0., scale=1.)

            # Variable for storing mean of weight matrix. Used by lr and pfp
            self.w_m = tf.get_variable(name='w_m', shape=self.w_shape,
                                       initializer=get_initializer(self.layer_config['init_scheme']['w'],
                                                                   self.w_shape))
            # Variable for storing variance of weight matrix. Used by lr and pfp
            self.w_v = tf.exp(tf.get_variable(name='w_v', shape=self.w_shape,
                                              initializer=get_initializer(self.layer_config['init_scheme']['w_v'],
                                                                          self.w_shape)))
            # Variable for storing deterministic weight. Used by MC dropout
            self.w = tf.get_variable(name='w', shape=self.w_shape,
                                     initializer=get_initializer(self.layer_config['init_scheme']['w'], self.w_shape))

            # Variable for storing mean of bias. Used by lr and pfp
            self.b_m = tf.get_variable(name='b_m', shape=self.b_shape,
                                       initializer=get_initializer(self.layer_config['init_scheme']['b'],
                                                                   self.b_shape))
            # Variable for storing variance of bias. Used by lr and pfp
            self.b_v = tf.exp(tf.get_variable(name='b_v', shape=self.b_shape,
                                              initializer=get_initializer(self.layer_config['init_scheme']['b_v'],
                                                                          self.b_shape)))
            # Variable for storing deterministic bias. Used by MC dropout
            self.b = tf.get_variable(name='b', shape=self.b_shape,
                                     initializer=get_initializer(self.layer_config['init_scheme']['b'], self.b_shape))

    # Creates a probabilistic forward pass
    # First mean and variance of activation is calculated. Then the distribution of the output of the layer is
    # approximated with a Gaussian using moment matching.
    def create_pfp(self, x_m, x_v):
        a_m, a_v = approx_activation(self.w_m, self.w_v, self.b_m, self.b_v, x_m, x_v)

        if self.layer_config['act_func'] == 'linear':
            return a_m, a_v
        elif self.layer_config['act_func'] == 'sig':
            return transform_sig_activation(a_m, a_v)
        elif self.layer_config['act_func'] == 'tanh':
            return transform_tanh_activation(a_m, a_v)
        else:
            raise Exception('Activation function {} not supported'.format(self.layer_config['act_func']))

    # Creates a forward pass using the local reparametrization trick. Mean and standard deviation of the distribution of
    # activations is calculated, then sampled from, and the sample is passed through the activation function
    def create_lr_fp(self, x):
        mean = tf.matmul(x, self.w_m) + self.b_m
        std = tf.sqrt(tf.matmul(tf.square(x), self.w_v) + self.b_v)
        shape = (tf.shape(x)[0], tf.shape(self.b_m)[1])
        a = mean + tf.multiply(self.gauss.sample(sample_shape=shape), std)
        return self.apply_act_func(a)

    # Creates a forward pass with dropout
    def create_fp(self, x):
        a = tf.matmul(x, self.w) + self.b
        return tf.nn.dropout(x=self.apply_act_func(a), keep_prob=self.layer_config['keep_prob'])

    def apply_act_func(self, a):
        if self.layer_config['act_func'] == 'linear':
            return a
        elif self.layer_config['act_func'] == 'sig':
            return tf.nn.sigmoid(a)
        elif self.layer_config['act_func'] == 'tanh':
            return tf.nn.tanh(a)
        else:
            raise Exception('Activation function {} not supported'.format(self.layer_config['act_func']))



