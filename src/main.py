import sys
import os
import copy

sys.path.append('../')

from src.experiment import Experiment
from src.nn_data import save_to_file

try:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    print('NO SLURM TASK ID FOUND')
    print('Using ID 0 instead')
    task_id = 0

filename = 'ag_test'

runs = 1

data_config = {'dataset': 'ag55',
               'tr': {'minibatch_enabled': True,
                      'minibatch_size': 1000},
               'va': {'minibatch_enabled': True,
                      'minibatch_size': 1000},
               'ca': {'minibatch_enabled': True,
                      'minibatch_size': 1000}}

# TODO: Think about initialization... for pfp and lr pretraining with deterministic NN may be good to find proper initialization for variance
hidden_1_config = {'var_scope': 'hidden_1',
                   'layer_type': 'fc',
                   'act_func': 'tanh',
                   'keep_prob': 0.8}

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

ag_nn_config = {'layout': [19, 40, 40, 1],
                'var_scope': 'ag',
                'layer_configs': [{'layer_type': 'input'}, hidden_1_config, hidden_2_config, output_config]}

nn_config = {'data_m': 1.}
nn_config['atoms'] = ['ag'] * 55
nn_config['ag'] = ag_nn_config

train_config = {'learning_rate': 0.002,
                'max_epochs': 100,
                'min_error': 0.,
                'beta': 0.1,
                'method': 'lr',
                'pretrain': {'enabled': False, 'path': '../models/pretrained'},
                'task_id': task_id}

candidate_config = {}

info_config = {'calc_performance_every': 1,
               'n_samples': 10,
               'filename': filename,
               'tensorboard': {'enabled': False, 'path': '../tb/' + filename, 'period': 1,
                               'weights': True, 'gradients': True, 'results': True, 'acts': True},
               'profiling': {'enabled': False, 'path': '../profiling/' + filename},
               'record_metrics': ['tr', 'va', 'ca'],
               'record_output': ['ca']}

result_config = {'save_results': True,
                 'path': '../numerical_results/' + filename + '_' + str(task_id),
                 'plot_results': False,
                 'print_final_stats': True}

nn_datas = []
for run in range(runs):
    experiment = Experiment(nn_config, train_config, data_config, candidate_config, info_config)
    nn_datas.append(experiment.train())
print('----------------------------')
if result_config['save_results']:
    save_to_file(nn_datas, result_config['path'])

