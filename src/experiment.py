import tensorflow as tf
from tensorflow.python.client import timeline
from src.data_loader import load
from src.datasets import DatasetContainer
from src.nn import NN


class Experiment:
    def __init__(self, nn_config, train_config, data_config, candidate_config, info_config):
        self.nn_config = nn_config
        self.train_config = train_config
        self.candidate_config = candidate_config
        self.data_config = data_config
        self.info_config = info_config
        self.nn = None

    def train(self):
        max_epochs = self.train_config['max_epochs']
        min_error = self.train_config['min_error']

        data_dict = load(self.data_config['dataset'])
        datasets = DatasetContainer(self.data_config, data_dict)
        datasets.transfer_samples('ca', 'tr', [1, 5, 2])
        self.nn = NN(self.nn_config, self.train_config, self.info_config, datasets)

        #  Trained NN is optionally stored and can later be loaded with the pretrain option
        model_path = '../models/' + self.info_config['filename'] + '_' + str(self.train_config['task_id'])
        model_saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as sess:
            if self.info_config['profiling']['enabled']:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            else:
                options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
            run_metadata = tf.RunMetadata()
            writer = tf.summary.FileWriter(self.info_config['tensorboard']['path'] + str(self.train_config['task_id']))
            sess.run(tf.global_variables_initializer())

            # Loads previously stored weights
            if self.train_config['pretrain']['enabled']:
                model_saver.restore(sess, self.train_config['pretrain']['path'])

            # Loading datasets into GPU
            for key in data_dict.keys():
                sess.run(datasets.sets[key]['load'],
                         feed_dict={datasets.sets[key]['x_ph']: data_dict[key]['x'],
                                    datasets.sets[key]['t_ph']: data_dict[key]['t']})
            print('what')
            print(sess.run(datasets.test).shape)

            for epoch in range(max_epochs):
                # Evaluate performance on the different datasets and print some results on console
                # Also check stopping critera
                if epoch % self.info_config['calc_performance_every'] == 0:
                    self.nn.nn_data.retrieve_metrics(sess, epoch)
                    self.nn.nn_data.print_metrics()
                    if self.nn.nn_data.metric_dict['tr']['vfe'][-1] < min_error:
                        break

                # Optionally store profiling results of this epoch in files
                if self.info_config['profiling']['enabled']:
                    for trace_idx, trace in enumerate(traces):
                        path = self.info_config['profiling']['path'] + '_' + str(epoch) + '_' + str(trace_idx)
                        with open(path + 'training.json', 'w') as f:
                            f.write(trace)

                # Optionally store tensorboard summaries (not fully implemented yet)
                # TODO: Implement summaries
                if self.info_config['tensorboard']['enabled'] \
                        and epoch % self.info_config['tensorboard']['period'] == 0:
                    if self.info_config['tensorboard']['weights']:
                        weight_summary = sess.run(self.rnn.weight_summaries,
                                                  feed_dict={datasets.minibatch_idx: 0})
                        writer.add_summary(weight_summary, epoch)
                    if self.info_config['tensorboard']['gradients']:
                        gradient_summary = sess.run(self.nn.gradient_summaries,
                                                    feed_dict={datasets.minibatch_idx: 0})
                        writer.add_summary(gradient_summary, epoch)
                    if self.info_config['tensorboard']['results']:
                        metric_summaries = sess.run(self.nn.metric_summaries,
                                                    feed_dict={datasets.minibatch_idx: 0})
                        writer.add_summary(metric_summaries, epoch)
                    if self.info_config['tensorboard']['acts']:
                        act_summaries = sess.run(self.nn.act_summaries, feed_dict={datasets.minibatch_idx: 0})
                        writer.add_summary(act_summaries, epoch)

                # Train for one full epoch. First shuffle to create new minibatches from the given data and
                # then do a training step for each minibatch.
                sess.run(datasets.sets['tr']['shuffle'])
                traces = list()
                for minibatch_idx in range(datasets.sets['tr']['n_minibatches']):
                    sess.run(self.nn.train_op,
                             feed_dict={datasets.minibatch_idx: minibatch_idx},
                             options=options, run_metadata=run_metadata)
                traces.append(timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())

            model_saver.save(sess, model_path)
        writer.close()
        return self.nn.nn_data


