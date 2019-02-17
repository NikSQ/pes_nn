# This class handles additional data about the NN called 'metrics' and 'output'
# metrics: Performance metrics of NN (loss, variational free energy, kl loss, expected log likelihood)
# output: Mean and variance of output of NN for a given dataset
import numpy as np
import tensorflow as tf

def save_to_file(nn_datas, path):
    metric_dicts = []
    for nn_data in nn_datas:
        metric_dicts.append(nn_data.metric_dict)

    for data_key in metric_dicts[0].keys():
        if data_key == 'epoch':
            continue

        metrics = convert_to_array(metric_dicts, data_key)

        for metric_key in metric_dicts[0][data_key].keys():
            np.save(path + '_' + data_key + '_' + metric_key, metrics[metric_key])
    np.save(path + '_epochs', np.asarray(metric_dicts[0]['epoch']))

    # TODO: Implement computation of mean and variance over different nn_data (just one supported atm)
    #ordered_vars = np.concatenate(nn_datas[0].var_dict['ordered'], axis=1)
    #random_vars = np.concatenate(nn_datas[0].var_dict['random'], axis=1)
    #epochs = nn_datas[0].var_dict['epoch']
    #np.save(path + '_v_ordered', ordered_vars)
    #np.save(path + '_v_random', random_vars)
    #np.save(path + '_v_epochs', epochs)


def convert_to_array(metric_dicts, data_key):
    metrics = dict()
    metric_keys = metric_dicts[0][data_key].keys()
    for metric_key in metric_keys:
        metrics[metric_key] = list()

    for metric_dict in metric_dicts:
        for metric_key in metric_keys:
            metrics[metric_key].append(np.expand_dims(np.asarray(metric_dict[data_key][metric_key], np.float32),
                                       axis=0))

    for metric_key in metric_keys:
        metrics[metric_key] = np.concatenate(metrics[metric_key])

    return metrics


class NNData:
    def __init__(self, l_data, info_config):
        self.info_config = info_config
        self.l_data = l_data
        self.metric_dict = {'epoch': list()}
        self.metric_ops = None
        self.output_dict = dict()
        self.output_ops = None
        self.var_dict = {'ordered': [], 'random': [], 'epoch': []}

    # Adds operations which calculate metrics for a dataset, given by its respective data_key
    def add_metrics(self, vfe_op, kl_op=None, elogl_op=None):
        for data_key in self.info_config['record_metrics']:
            self.metric_dict.update({data_key: {'vfe': []}})
        self.metric_ops = vfe_op

    # Adds operations which calculate the output for a dataset, given by its respective data_key
    def add_output(self, mean_op, var_op):
        for data_key in self.info_config['record_output']:
            self.output_dict.update({data_key: {'ind_means': None, 'mean': None, 'var': None}})
        self.output_ops = [mean_op, var_op]

    # Retrieves metrics of the performance of the datasets
    def retrieve_metrics(self, sess, epoch):
        for data_key in self.info_config['record_metrics']:
            results = self.l_data.run(sess, self.metric_ops, data_key, shuffle=False, feed_t=True)

            cum_vfe = 0
            for vfe in results:
                cum_vfe += vfe
            vfe = cum_vfe / len(results)

            self.metric_dict[data_key]['vfe'].append(vfe)
        self.metric_dict['epoch'].append(epoch)


    # Calculates output (mean and variance) of given datasets
    # In case of pfp a single pass is made, in case of local reparameterization and MC dropout a hyperparameter controls
    # the amount of NN samples used to calculate expectation and variance.
    # n_samples must be None iff pfp was used.
    def retrieve_output(self, sess, epoch, n_samples=None):
        for data_key in self.info_config['record_output']:
            if n_samples is None:
                results = self.l_data.run(sess, self.output_ops, data_key, shuffle=False, feed_t=False,
                                          is_training=True)

                mean_list = []
                variance_list = []
                for mean, variance in results:
                    mean_list.append(mean)
                    variance_list.append(variance)

                self.output_dict[data_key]['mean'].append(np.concatenate(mean_list, axis=0))
                self.output_dict[data_key]['var'].append(np.concatenate(variance_list, axis=0))
            else:
                output_list = []
                for nn_sample_idx in range(n_samples):
                    results = self.l_data.run(sess, self.output_ops, data_key, shuffle=False, feed_t=False,
                                              is_training=True)

                    single_output = []
                    for mean, variance in results:
                        single_output.append(mean)
                    output_list.append(np.expand_dims(np.concatenate(single_output, axis=0), axis=1))
                output = np.concatenate(output_list, axis=1)
                self.output_dict[data_key]['ind_means'] = output
                self.output_dict[data_key]['mean'] = np.mean(output, axis=1)
                self.output_dict[data_key]['var'] = np.var(output, axis=1, ddof=1)


    # Prints the some of the latest metrics of the performance on training and validation set
    def print_metrics(self):
        print('{:3} | tr vfe: {:6.4f}, Va vfe: {:8.5f}'
              .format(self.metric_dict['epoch'][-1], self.metric_dict['tr']['vfe'][-1],
                      self.metric_dict['va']['vfe'][-1]))
