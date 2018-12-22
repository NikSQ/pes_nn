import sys
import os
import copy

sys.path.append('../')

from src.experiment import Experiment

try:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    print('NO SLURM TASK ID FOUND')
    print('Using ID 0 instead')
    task_id = 0

filename = 'ag_test'

runs = 1

data_config = {'dataset': 'ag50',
               'tr': {'minibatch_enabled': True,
                      'minibatch_size': 1000},
               'va': {'minibatch_enabled': True,
                      'minibatch_size': 1000}}

hidden_1_config = {'var_scope': 'hidden_1',
                   'layer_type': 'fc',
                   'act_func': 'tanh',
                   'init_scheme': {'w_m': 'xavier',
                                   'w_v': 0.01,
                                   'w': 'xavier',
                                   'b_m': 'zeros',
                                   'b_v': 0.01,
                                   'b': 'zeros'}}

hidden_2_config = copy.deepcopy(hidden_1_config)
hidden_2_config['var_scope'] = 'hidden_2'
output_config = copy.deepcopy(hidden_1_config)
output_config['var_scope'] = 'output'
output_config['act_func'] = 'linear'

atomic_nn_config = {'layout': [19, 40, 40, 1],
                    'layer_configs': [hidden_1_config, hidden_2_config, output_config],
                    'data_m': 1.}


nn_config = {'data_m': 1.}

nn_config['atomic_configs'] = [atomic_nn_config] * 55

training_config = {'learning_rate': 0.002,
                   'max_epochs': 100,
                   'type': 'l_sampling',
                   'is_pretrain': False,
                   'var_reg': 0,
                   'beta_reg': .1,
                   'mode': {'name': 'inc_lengths',
                            'in_seq_len': [1, 2, 4, 8, 30],
                            'out_seq_len': [1, 1, 2, 4, 5],
                            'zero_padding': [0, 0, 0, 2, 23],
                            'min_errors': [0., 0., 0., 0., 0.],
                            'max_epochs': [2, 10, 30, 50, 1000]},
                   'task_id': task_id}

training_config['mode'] = {'name': 'classic', 'min_error': 0., 'max_epochs': 5}
pretraining_config = copy.deepcopy(training_config)
pretraining_config['is_pretrain'] = True
pretraining_config['reg'] = 1.


info_config = {'calc_performance_every': 1,
               'n_samples': 10,
               'tensorboard': {'enabled': False, 'path': '../tb/' + filename, 'period': 1,
                               'weights': True, 'gradients': True, 'results': True, 'acts': True},
               'profiling': {'enabled': False, 'path': '../profiling/' + filename}}

result_config = {'save_results': True,
                 'path': '../numerical_results/' + filename + '_' + str(task_id),
                 'plot_results': False,
                 'print_final_stats': True}

result_dicts = []
for run in range(runs):
    experiment = Experiment()
    result_dicts.append(experiment.train(rnn_config, labelled_data_config, training_config, pretraining_config, info_config))
print('----------------------------')
if result_config['save_results']:
    save_to_file(result_dicts, result_config['path'])

