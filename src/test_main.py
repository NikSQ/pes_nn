import sys
import os
import copy
import numpy as np

sys.path.append('../')

from src.experiment import Experiment

from src.nn_data import save_to_file

try:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    print('NO SLURM TASK ID FOUND')
    task_id = 0
    print('Using ID {} instead'.format(task_id))

filename_old = 'agauss_p50_l09'
filename = filename_old + '_test'

runs = 4

# data_config contains configuration about data
# dataset: name of the dataset as used in data_loader.py
# each part of the dataset has a data_key. Those are 'tr' for train, 'va' for validation, 'te' for test, 'ca' for candidate.
# disabling minibatchmode for training and candidate set means that the input matrix changes shapes each time samples
# are transferred from candidate to training set
data_config = {'dataset': 'ag55',
               'tr': {'minibatch_enabled': False,
                      'minibatch_size': 20},
               'va': {'minibatch_enabled': False,
                      'minibatch_size': 1000},
               'ca': {'minibatch_enabled': False,
                      'minibatch_size': 1000},
               'te': {'minibatch_enabled': False,
                      'minibatch_size': 1000}}

# TODO: More sophisticated initialization of variance
# Each layer has a layer_config. It is used in fc_layer.py
# var_scope: defines the variable scope within tensorflow
# layer_type: only 'fc' is supported here (fc = fully connected)
# act_func: activation function. at the moment the functions 'tanh', 'sig' and 'linear' are possible
# keep_prob: keep probabilities for dropout. Is only required of training method is MC dropout
# init_scheme: method of initialization for variables. (see fc_layer.py)
hidden_1_config = {'var_scope': 'hidden_1',
                   'layer_type': 'fc',
                   'act_func': 'tanh',
                   'keep_prob': .9,
                   'gauss_std': .0,
                   'dropout': 'ber',
                   'init_scheme': {'w_m': 'xavier',
                                   'w_v': 0.1,
                                   'w': 'xavier',
                                   'b_m': 'zeros',
                                   'b_v': 0.1,
                                   'b': 'zeros'}}
print('dropout: {}, prob: {}, std: {}'.format(hidden_1_config['dropout'], hidden_1_config['keep_prob'], hidden_1_config['gauss_std']))
hidden_2_config = copy.deepcopy(hidden_1_config)
hidden_2_config['var_scope'] = 'hidden_2'
output_config = copy.deepcopy(hidden_1_config)
output_config['var_scope'] = 'output'
output_config['act_func'] = 'linear'
output_config['init_scheme']['b'] = -2.1
output_config['init_scheme']['b_m'] = -2.1
output_config['dropout'] = 'ber'
output_config['keep_prob'] = 1.  # Needs to be done explicitly

# Each atom nn has a config. Will usually be the same for each atom though, except for the var_scope
# layout: Size of each layer (including input and output layer)
# layer_configs: The layer configuration dicts for each layer
ag_nn_config = {'layout': [54, 40, 40, 1],
                'var_scope': 'ag',
                'layer_configs': [{'layer_type': 'input'}, hidden_1_config, hidden_2_config, output_config]}

# Config dict of the complete network
# data_m: Variable used to trade-off KL loss and expected log likelihood. data_m can be interpreted as a factor with
# which the training data is multiplied with, to increase its influence over the prior. Not used in MC dropout
# atoms: list of atoms. need to appear in same order as in the dataset. the atom names must be the same as the var_scope
# used for the atom-nn-config
# for each different atom type, the nn-config stores the atom-nn-config using the var_scope of the atom-nn-config as
# dict_key. In this case 'ag'
nn_config = {'data_m': 10000000.}
nn_config['atoms'] = ['ag'] * 55
nn_config['ag'] = ag_nn_config
nn_config['l2'] = 0.
print('l2: {}'.format(nn_config['l2']))
# This is the configuration for training related settings
# learning_rate: initial learning rate of the adam optimizer
# max_epochs: maximum amount of epochs the nn is trained
# min_error: alternative stopping    criterion. as soon as loss or variational free energy falls below it, the training
# stops
# beta: Used by the probabilistic forward pass. It specifies the variance of the output of a deterministic NN
# out_var: Specifies variance of output of a deterministc NN.
# the pfp requires the network to output a probability density instead of a single energy.
# method: training method for the NN. 'lr' - local reparametrization, 'pfp' - probabilistic forward pass, 'mcd': mc dropout
# pretrain: if enabled, the network is initialized with weights stored in the given path
train_config = {'learning_rate': .00008,
                'max_epochs': 20,
                'min_error': 0.,
                'out_var': 0.2,
                'w_prior_v': 0.6,
                'method': 'mcd',
                'pretrain': {'enabled': False, 'path': '../models/normal_0', 'init_mcd': True},
                'task_id': task_id}
print(train_config)

# period: after how many epochs the candidates are evaluated and partly transferred to training set
# n_transfers: how many candidates are transferred to training set each time
# n_samples: specifies how many forward passes should be made to estimate variance (not used by pfp)
methods = ['std', 'random', 'entropy']
candidate_config = {'period': 50000,
                    'method': methods[task_id],
                    'n_transfers': 1,
                    'n_samples': 3}
print(candidate_config)


info_config = {'calc_performance_every': 1,
               'filename': filename,
               'tensorboard': {'enabled': False, 'path': '../tb/' + filename, 'period': 200, 'graph': False,
                               'weights': True, 'gradients': True},
               'profiling': {'enabled': False, 'path': '../profiling/' + filename},
               'record_metrics': ['tr', 'va', 'ca', 'te'],
               'record_output': ['ca']}

result_config = {'save_results': True,
                 'path': '../numerical_results/' + filename + '_' + str(task_id),
                 'plot_results': False,
                 'print_final_stats': True}


nn_datas = []
transfer_idcs = []
tr_idcs = []
filename_old = filename
transfer_idcs = np.load('../numerical_results/' + filename_old + '_' + str(task_id) + '_transferidcs.npy')
tr_idcs = np.load('../numerical_results/' + filename_old + '_' + str(task_id) + '_tr_idcs.npy')
tr_set_idcs = np.concatenate([transfer_idcs, tr_idcs], axis=1)
for run in range(runs):
    experiment = Experiment(data_config, candidate_config, info_config)
    nn_data, u = experiment.train(transfer_idc=tr_set_idcs[run, :], nn_config=nn_config, train_config=train_config)
    nn_datas.append(nn_data)

if result_config['save_results']:
    save_to_file(nn_datas, result_config['path'])

