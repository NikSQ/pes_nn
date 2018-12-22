# This class handles additional data about the NN called 'metrics' and 'output'
# metrics: Performance metrics of NN (loss, variational free energy, kl loss, expected log likelihood)
# output: Mean and variance of output of NN for a given dataset
import numpy as np


def save_to_file(result_dicts, path):
    for process_key in result_dicts[0].keys():
        if process_key == 'epoch':
            continue

        if process_key.endswith('_s'):
            metrics = convert_to_array(result_dicts, process_key, ['m_loss', 's_loss', 'm_acc', 's_acc'])
        elif process_key.endswith('_b'):
            metrics = convert_to_array(result_dicts, process_key, ['nelbo', 'kl', 'elogl', 'acc'])
        else:
            raise Exception('process key {} not understood'.format(process_key))

        for metric_key in metrics.keys():
            np.save(path + '_' + process_key + '_' + metric_key, metrics[metric_key])
    np.save(path + '_epochs', np.asarray(result_dicts[0]['epoch']))


def convert_to_array(result_dicts, process_key, metric_keys):
    metrics = dict()
    for metric_key in metric_keys:
        metrics[metric_key] = list()

    for run, result_dict in enumerate(result_dicts):
        for metric_key in metric_keys:
            metrics[metric_key].append(np.expand_dims(np.asarray(result_dict[process_key][metric_key], np.float32),
                                       axis=0))

    for metric_key in metric_keys:
        metrics[metric_key] = np.concatenate(metrics[metric_key])

    return metrics


class NNData:
    def __init__(self, data_config, datasets, info_config):
        self.info_config = info_config
        self.data_config = data_config
        self.datasets = datasets
        self.metric_dict = {'epoch': list()}
        self.metric_op_dict = {}
        self.output_dict = {'epoch': list()}
        self.output_op_dict = {}

    # Adds operations which calculate metrics for a dataset, given by its respective data_key
    def add_metrics(self, data_key, vfe_op, kl_op, elogl_op):
        if data_key in self.info_config['record_metrics']:
            self.metric_dict.update({data_key: {'vfe': [], 'kl': [], 'elogl': []}})
            self.metric_op_dict.update({data_key: [vfe_op, kl_op, elogl_op]})

    # Adds operations which calculate the output for a dataset, given by its respective data_key
    def add_output(self, data_key, mean_op, var_op):
        if data_key in self.info_config['record_output']:
            self.output_dict.update({data_key: {'mean': [], 'var': []}})
            self.output_op_dict.update({data_key: [mean_op, var_op]})

    # Retrieves metrics of the performance of the datasets
    def retrieve_metrics(self, sess, epoch):
        for data_key in self.metric_op_dict.keys():
            cum_vfe = 0
            elogl = 0
            for minibatch_idx in range(self.datasets.data[data_key]['n_minibatches']):
                vfe, kl, elogl = sess.run(self.metric_op_dict[data_key],
                                          feed_dict={self.datasets.batch_idx: minibatch_idx})
                cum_vfe += vfe
                elogl += elogl
            nelbo = cum_vfe / self.datasets.data[data_key]['n_minibatches']

            self.metric_dict[data_key]['vfe'].append(nelbo)
            self.metric_dict[data_key]['kl'].append(kl)
            self.metric_dict[data_key]['elogl'].append(elogl)
        self.metric_dict['epoch'].append(epoch)

    # Calculates output (mean and variance) of given datasets
    # In case of pfp a single pass is made, in case of local reparameterization and MC dropout a hyperparameter controls
    # the amount of NN samples used to calculate expectation and variance.
    # n_samples must be None iff pfp was used.
    def retrieve_output(self, sess, epoch, n_samples=None):
        for data_key in self.output_op_dict.keys():
            if n_samples is None:
                mean_list = []
                variance_list = []
                for minibatch_idx in range(self.datasets.data[data_key]['n_minibatches']):
                    mean, variance = sess.run(self.output_op_dict[data_key],
                                              feed_dict={self.datasets.batch_idx: minibatch_idx})
                    mean_list.append(mean)
                    variance_list.append(variance)
                self.output_dict[data_key]['mean'].append(np.concatenate(mean_list, axis=0))
                self.output_dict[data_key]['var'].append(np.concatenate(variance_list, axis=0))
            else:
                # Elements of output_list store output of complete batch;
                output_list = []
                for nn_sample_idx in range(n_samples):
                    # Elements of single_output contain output of minibatch, the list contains whole output of one
                    # sample operation
                    single_output = []
                    for minibatch_idx in range(self.datasets.data[data_key]['n_minibatches']):
                        output = sess.run(self.output_op_dict[data_key],
                                                  feed_dict={self.datasets.batch_idx: minibatch_idx})
                        single_output.append(output)
                    output_list.append(np.expand_dims(np.concatenate(single_output, axis=0), axis=1))

                self.output_dict[data_key]['mean'].append(np.mean(output_list, axis=1))
                self.output_dict[data_key]['var'].append(np.var(output_list, axis=1, ddof=1))

        self.output_dict['epoch'].append(epoch)

    # Prints the some of the latest metrics of the performance on training and validation set
    def print_metrics(self):
        print('{:3} | tr vfe: {:6.4f}, Va vfe: {:8.5f}'
              .format(self.metric_dict['epoch'][-1], self.metric_dict['tr']['vfe'][-1],
                      self.metric_dict['va']['vfe'][-1]))
