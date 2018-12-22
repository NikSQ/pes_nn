import numpy as np
import tensorflow as tf

# TODO: Add operation for moving a sample from candidate to training dataset
# TODO: Allow option to take the gradient w.r.t. to input of training and / or candidate set
# TODO: Add symmetry operators (descriptors)

class DatasetContainer:
    def __init__(self, data_config, data_dict):
        with tf.variable_scope('datasets'):
            self.batch_idx = tf.placeholder(dtype=tf.int32)  # indexes a minibatch within the larger batch
            self.data_config = data_config
            self.sets = dict()

            # the keys refer to the different datasets (training, validation and / or test)
            with tf.scope('datasets'):
                for data_key in data_dict.keys():
                    dataset = dict()
                    dataset['x_shape'] = data_dict[data_key]['x'].shape
                    dataset['x_ph'] = tf.placeholder(dtype=tf.float32, shape=dataset['x_shape'])
                    dataset['x'] = tf.get_variable(name='x_' + data_key, shape=dataset['x_shape'], dtype=tf.float32,
                                                   trainable=False)
                    assign_x = tf.assign(dataset['x'], dataset['x_ph'])

                    # Candidate datasets only consist of input data
                    if data_key != 'ca':
                        dataset['t_shape'] = data_dict[data_key]['t'].shape
                        dataset['t_ph'] = tf.placeholder(dtype=tf.float32, shape=dataset['t_shape'])
                        dataset['t'] = tf.get_variable(name='t_' + data_key, shape=dataset['t_shape'], dtype=tf.float32,
                                                       trainable=False)
                        assign_t = tf.assign(dataset['t'], dataset['t_ph'])
                        dataset['load'] = tf.group(*[assign_x, assign_t])
                    else:
                        dataset['load'] = assign_x

                    if data_config[data_key]['minibatch_enabled']:
                        batch_size = data_config[data_key]['minibatch_size']

                        # Number of samples is expanded, such that the number of samples is a multiple of the batch size
                        dataset['n_minibatches'] = int(np.ceil(float(dataset['x_shape'][0]) / float(batch_size)))
                        n_samples = batch_size * dataset['n_minibatches']

                        # A shuffled list of sample indices. Iterating over the complete list will be one epoch
                        sample_list = tf.get_variable(name=data_key + '_sample_list', shape=n_samples, dtype=tf.int32,
                                                      trainable=False)
                        samples = tf.tile(tf.random_shuffle(tf.range(dataset['x_shape'][0])),
                                          multiples=[int(np.ceil(n_samples / dataset['x_shape'][0]))])

                        # This op shuffles through the samples in the dataset.
                        dataset['shuffle'] = tf.assign(sample_list, samples[:n_samples])

                        dataset['x'] = tf.gather(dataset['x'],
                                                 indices=samples[self.batch_idx:self.batch_idx+batch_size])
                        dataset['x_shape'] = (data_config[data_key]['minibatch_size'],) + dataset['x_shape'][1:]

                        if data_key != 'ca':
                            dataset['t'] = tf.gather(dataset['t'],
                                                     indices=samples[self.batch_idx:self.batch_idx+batch_size])
                            dataset['t_shape'] = (data_config[data_key]['minibatch_size'],) + dataset['t_shape'][1:]
                    else:
                        dataset['n_minibatches'] = 1
                        dataset['shuffle'] = tf.no_op()

                    self.sets.update({data_key: dataset})
