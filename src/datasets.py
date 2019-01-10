import numpy as np
import tensorflow as tf

# TODO: Allow option to take the gradient w.r.t. to input of training and / or candidate set
# TODO: Add symmetry operators (descriptors)


class LabeledData:
    def __init__(self, data_config, data_dict):
        self.data_dict = data_dict
        self.data_config = data_config
        self.update_shapes()

        with tf.variable_scope('datasets'):
            self.x = tf.placeholder(name='x', dtype=tf.float32,
                                    shape=(None,) + next(iter(data_dict.values()))['x'].shape[1:])
            self.t = tf.placeholder(name='t', dtype=tf.float64, shape=(None,))
            self.is_training = tf.placeholder(dtype=tf.bool)

    def update_shapes(self):
        for data_key in self.data_dict.keys():
            if self.data_config[data_key]['minibatch_enabled']:
                self.data_config[data_key]['n_minibatches'] = int(np.ceil(float(self.data_dict[data_key]['x'].shape[0]) /
                                                                          float(self.data_config[data_key]['minibatch_size'])))
                self.data_config[data_key]['batch_size'] = self.data_config[data_key]['n_minibatches'] * \
                                                           self.data_config[data_key]['minibatch_size']
            else:
                self.data_config[data_key]['n_minibatches'] = 1
                self.data_config[data_key]['batch_size'] = self.data_dict[data_key]['x'].shape[0]

    def transfer_samples(self, transfer_idc, src_data_key='ca', tgt_data_key='tr'):
        self.data_dict[tgt_data_key]['x'] = np.concatenate([self.data_dict[tgt_data_key]['x'],
                                                            self.data_dict[src_data_key]['x'][transfer_idc]], axis=0)
        self.data_dict[tgt_data_key]['t'] = np.concatenate([self.data_dict[tgt_data_key]['t'],
                                                            self.data_dict[src_data_key]['t'][transfer_idc]], axis=0)
        self.data_dict[src_data_key]['x'] = np.delete(self.data_dict[src_data_key]['x'], transfer_idc, axis=0)
        self.data_dict[src_data_key]['t'] = np.delete(self.data_dict[src_data_key]['t'], transfer_idc, axis=0)
        self.update_shapes()

    def create_minibatches(self, data_key, shuffle, feed_t, is_training):
        idc_list = np.arange(self.data_dict[data_key]['x'].shape[0])
        if shuffle:
            idc_list = np.random.permutation(idc_list)
        idc_list = np.tile(idc_list, reps=(int(np.ceil(self.data_config[data_key]['batch_size'] / len(idc_list)))))
        minibatches = []
        for minibatch_idx in range(self.data_config[data_key]['n_minibatches']):
            idc_range = range(minibatch_idx * self.data_config[data_key]['minibatch_size'],
                              (minibatch_idx+1) * self.data_config[data_key]['minibatch_size'])
            if feed_t:
                minibatch = {self.x: self.data_dict[data_key]['x'][idc_list[idc_range]],
                             self.t: self.data_dict[data_key]['t'][idc_list[idc_range]],
                             self.is_training: is_training}
            else:
                minibatch = {self.x: self.data_dict[data_key]['x'][idc_list[idc_range]],
                             self.is_training: is_training}

            minibatches.append(minibatch)
        return minibatches

    def run(self, sess, ops, data_key, shuffle=False, feed_t=True, is_training=False):
        if self.data_config[data_key]['minibatch_enabled']:
            results = []
            feed_dicts = self.create_minibatches(data_key, shuffle, feed_t, is_training)
            for minibatch_idx in range(self.data_config[data_key]['n_minibatches']):
                results.append(sess.run(ops, feed_dict=feed_dicts[minibatch_idx]))
            return results
        else:
            feed_dict = {self.x: self.data_dict[data_key]['x'],
                         self.is_training: is_training}
            if feed_t:
                feed_dict[self.t] = self.data_dict[data_key]['t']
            result = sess.run(ops, feed_dict=feed_dict)
            return [result]


# This class is obsolete for the moment
class DatasetContainer:
    def __init__(self, data_config, data_dict):
        with tf.variable_scope('datasets'):
            self.minibatch_idx = tf.placeholder(dtype=tf.int32)  # indexes a minibatch within the larger batch
            self.sl_indices = tf.placeholder(shape=(None,), dtype=tf.int32)
            self.data_config = data_config
            self.data = dict()

            # the keys refer to the different datasets (tr = training, va = validation, te = test, ca = candidate)
            for data_key in data_dict.keys():
                dataset = dict()
                dataset['x_f_shape'] = list(data_dict[data_key]['x'].shape)
                dataset['x_ph'] = tf.placeholder(dtype=tf.float32, shape=dataset['x_f_shape'])
                dataset['x_full'] = tf.get_variable(name='x_' + data_key, shape=dataset['x_f_shape'], dtype=tf.float32,
                                                    trainable=False)
                assign_x = tf.assign(dataset['x_full'], dataset['x_ph'])

                dataset['t_f_shape'] = list(data_dict[data_key]['t'].shape)
                dataset['t_ph'] = tf.placeholder(dtype=tf.float64, shape=dataset['t_f_shape'])
                dataset['t_full'] = tf.get_variable(name='t_' + data_key, shape=dataset['t_f_shape'], dtype=tf.float64,
                                                   trainable=False)
                assign_t = tf.assign(dataset['t_full'], dataset['t_ph'])
                dataset['load'] = tf.group(*[assign_x, assign_t])

                if data_config[data_key]['minibatch_enabled']:
                    minibatch_size = data_config[data_key]['minibatch_size']
                    dataset['minibatch_size'] = minibatch_size

                    # Number of samples is expanded, such that the number of samples is a multiple of the batch size
                    dataset['n_minibatches'] = int(np.ceil(float(dataset['x_f_shape'][0]) / float(minibatch_size)))

                    # A list of indices referring to samples of the batch (to x_full and t_full).
                    # Iterating over the complete list will be one epoch. Note that some indices will appear twice if
                    # batch_size is not a multiple of minibatch_size.
                    # The list can be shuffled using the shuffle op. This op is in particular of importance if some
                    # indices occur twice, as this op changes such 'privileged' samples.
                    sl_init_vals = np.arange(minibatch_size * dataset['n_minibatches'])
                    dataset['sample_list'] = tf.get_variable(name=data_key + '_sample_list',
                                                             shape=minibatch_size * dataset['n_minibatches'],
                                                             dtype=tf.int32,
                                                             trainable=False,
                                                             initializer=tf.constant_initializer(sl_init_vals))
                    dataset['assign_sl'] = tf.assign(dataset['sample_list'], self.sl_indices, validate_shape=False)
                    self.create_shuffle_op(dataset)

                    dataset['x'] = tf.gather(dataset['x_full'],
                                             indices=dataset['sample_list']
                                             [self.minibatch_idx:self.minibatch_idx+minibatch_size])
                    dataset['x_shape'] = [data_config[data_key]['minibatch_size'], dataset['x_f_shape'][1:]]

                    dataset['t'] = tf.gather(dataset['t_full'],
                                             indices=dataset['sample_list']
                                             [self.minibatch_idx:self.minibatch_idx+minibatch_size])
                    dataset['t_shape'] = [data_config[data_key]['minibatch_size'], dataset['t_f_shape'][1:]]
                else:
                    dataset['n_minibatches'] = 1
                    dataset['shuffle'] = tf.no_op()
                    dataset['x'] = dataset['x_full']
                    dataset['t'] = dataset['t_full']
                    dataset['x_shape'] = dataset['x_f_shape']
                    dataset['t_shape'] = dataset['t_f_shape']
                self.sets.update({data_key: dataset})

        # Create op which transfers samples from ca to tr dataset
        tgt_set = 'tr'
        src_set = 'ca'
        self.transfer_idc = tf.placeholder(shape=None, dtype=tf.int32)
        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32)
        op1 = tf.assign(self.sets[tgt_set]['x_full'], tf.concat([self.sets[tgt_set]['x_full'],
                                                                 tf.gather(self.sets[src_set]['x_full'],
                                                                           indices=self.transfer_idc)], axis=0),
                        validate_shape=False)
        op2 = tf.assign(self.sets[tgt_set]['t_full'], tf.concat([self.sets[tgt_set]['t_full'],
                                                                 tf.gather(self.sets[src_set]['t_full'],
                                                                           indices=self.transfer_idc)], axis=0),
                        validate_shape=False)
        with tf.control_dependencies([op1, op2]):
            c_idc = tf.squeeze(tf.setdiff1d(tf.range(self.batch_size), self.transfer_idc))[0]
            op3 = tf.assign(self.sets[src_set]['x_full'], tf.gather(self.sets[src_set]['x_full'], indices=c_idc),
                            validate_shape=False)
            op4 = tf.assign(self.sets[src_set]['t_full'], tf.gather(self.sets[src_set]['t_full'], indices=c_idc),
                            validate_shape=False)
            self.cidx = c_idc
        self.transfer_op = tf.group(*[op1, op2, op3, op4])

    def create_shuffle_op(self, dataset):
        n_samples = dataset['minibatch_size'] * dataset['n_minibatches']  # amount of samples in one epoch
        samples = tf.tile(tf.random_shuffle(tf.range(dataset['x_f_shape'][0])),
                          multiples=[int(np.ceil(n_samples / dataset['x_f_shape'][0]))])
        dataset['shuffle'] = tf.assign(dataset['sample_list'], samples[:n_samples], validate_shape=False)

    # Whenever samples are transferred from ca to tr, this function needs to be called.
    # Training set needs to be shuffled afterwards
    def update_shapes(self, n_transfers, sess):
        for data_key, n_change in zip(['tr', 'ca'], [n_transfers, -n_transfers]):
            dataset = self.sets[data_key]
            if self.data_config[data_key]['minibatch_enabled']:
                new_batch_size = dataset['x_f_shape'][0] + n_change
                minibatch_size = dataset['minibatch_size']
                dataset['n_minibatches'] = int(np.ceil(float(new_batch_size) /
                                                       float(minibatch_size)))
                if data_key != 'ca':
                    self.create_shuffle_op(dataset)
                else:
                    sess.run(dataset['assign_sl'], feed_dict={self.sl_indices: np.arange(dataset['minibatch_size'] *
                                                                                         dataset['n_minibatches'])})
            else:
                self.sets[data_key]['x_shape'][0] += n_change
                self.sets[data_key]['t_shape'][0] += n_change

            self.sets[data_key]['x_f_shape'][0] += n_change
            self.sets[data_key]['t_f_shape'][0] += n_change

