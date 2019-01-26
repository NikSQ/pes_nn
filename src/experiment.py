import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
from src.data_loader import load
from src.datasets import LabeledData
from src.nn import NN


class Experiment:
    def __init__(self, nn_config, train_config, data_config, candidate_config, info_config):
        tf.reset_default_graph()
        self.nn_config = nn_config
        self.train_config = train_config
        self.candidate_config = candidate_config
        self.data_config = data_config
        self.info_config = info_config
        self.nn = None
        self.l_data = None

    def train(self):
        max_epochs = self.train_config['max_epochs']
        min_error = self.train_config['min_error']

        data_dict = load(self.data_config['dataset'])
        l_data = LabeledData(self.data_config, data_dict)
        self.l_data = l_data
        self.nn = NN(self.nn_config, self.train_config, self.info_config, l_data)

        #  Trained NN is optionally stored and can later be loaded with the pretrain option
        model_path = '../models/' + self.info_config['filename'] + '_' + str(self.train_config['task_id'])
        model_saver = tf.train.Saver(tf.trainable_variables())

        var_summaries = []
        for variable in tf.trainable_variables():
            var_summaries.append(tf.summary.histogram(variable.name, variable))
        var_summaries = tf.summary.merge(var_summaries)

        with tf.Session() as sess:
            # if self.info_config['profiling']['enabled']:
            #     options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # else:
            #     options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
            # run_metadata = tf.RunMetadata()
            writer = tf.summary.FileWriter(self.info_config['tensorboard']['path'] + str(self.train_config['task_id']))
            sess.run(tf.global_variables_initializer())

            # Loads previously stored weights
            if self.train_config['pretrain']['enabled']:
                model_saver.restore(sess, self.train_config['pretrain']['path'])
                if self.train_config['pretrain']['init_mcd']:
                    # TODO: Atm this only works for a single type of atomic NN
                    sess.run(self.nn.atomic_nns[0].init_op)

            for epoch in range(max_epochs):
                # Run through candidate set to estimate the information gain from them
                if (epoch+1) % self.candidate_config['period'] == 0:
                    if self.train_config['method'] == 'pfp':
                        n_samples = None
                    else:
                        n_samples = self.candidate_config['n_samples']

                    self.nn.nn_data.retrieve_output(sess, epoch, n_samples)
                    n_transfers = self.candidate_config['n_transfers']
                    sorted_candidate_idc = np.argsort(self.nn.nn_data.output_dict['ca']['var'][-1]
                                                      [:l_data.data_config['ca']['batch_size']])
                    random_candidate_idc = np.random.permutation(sorted_candidate_idc)

                    self.nn.nn_data.add_variances(sorted_candidate_idc[-n_transfers:],
                                                  random_candidate_idc[-n_transfers:], epoch)

                    if self.candidate_config['method'] == 'random':
                        transfer_idc = random_candidate_idc
                    elif self.candidate_config['method'] == 'entropy':
                        transfer_idc = np.argsort(self.get_predictive_entropy())
                    else:
                        transfer_idc = sorted_candidate_idc

                    l_data.transfer_samples(transfer_idc[-n_transfers:])

                # Evaluate performance on the different datasets and print some results on console
                # Also check stopping critera
                if epoch % self.info_config['calc_performance_every'] == 0:
                    self.nn.nn_data.retrieve_metrics(sess, epoch)
                    self.nn.nn_data.print_metrics()
                    if self.nn.nn_data.metric_dict['tr']['vfe'][-1] < min_error:
                        break

                # Optionally store profiling results of this epoch in files
                # if self.info_config['profiling']['enabled']:
                #     for trace_idx, trace in enumerate(traces):
                #         path = self.info_config['profiling']['path'] + '_' + str(epoch) + '_' + str(trace_idx)
                #         with open(path + 'training.json', 'w') as f:
                #             f.write(trace)

                # Optionally store tensorboard summaries (not fully implemented yet)
                if self.info_config['tensorboard']['enabled'] \
                        and epoch % self.info_config['tensorboard']['period'] == 0:
                    if self.info_config['tensorboard']['weights']:
                        weight_summary = sess.run(var_summaries)
                        writer.add_summary(weight_summary, epoch)
                    if self.info_config['tensorboard']['gradients']:
                        gradient_summaries = l_data.run(sess, self.nn.gradient_summaries, 'tr', shuffle=False, feed_t=True)
                        writer.add_summary(gradient_summaries[0], epoch)
                    if self.info_config['tensorboard']['graph']:
                        writer.add_graph(sess.graph)

                # Train one full epoch
                l_data.run(sess, self.nn.train_op, 'tr', shuffle=True, feed_t=True, is_training=True)

            model_saver.save(sess, model_path)
        writer.close()
        return self.nn.nn_data

    def get_predictive_entropy(self):
        means = self.nn.nn_data.output_dict['ca']['ind_means'][:self.l_data.data_config['ca']['batch_size'], :]
        var = self.train_config['out_var']
        recip_var = 1 / (8 * var)
        entropies = np.zeros((means.shape[0],))
        for sample_i in range(means.shape[1]):
            xi = []
            for sample_j in range(sample_i, means.shape[1]):
                factor = 1 if sample_i == sample_j else 2
                xi.append(np.expand_dims(factor * recip_var * np.square(means[:, sample_i] - means[:, sample_j]),
                                         axis=1))
            xi = np.concatenate(xi, axis=1)
            m = np.max(xi, axis=1)
            entropies += m + np.log(np.sum(np.exp(xi - m[:, None])))
        print(entropies)
        return entropies






