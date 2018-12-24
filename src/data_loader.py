import numpy as np

path = '../datasets/'
sets = {'ag55': 'Ag_dataset.npz'}


def load(name):
    data = np.load(path + sets[name])
    labels = data['energies']
    features = data['Gs']

    # This is just used for testing the implementation
    tr_size = 1000
    va_size = 1000
    # all other samples are candidates

    data_dict = {'tr': dict(), 'va': dict(), 'ca': dict()}
    data_dict['tr']['x'] = features[:tr_size, :, :]
    data_dict['tr']['t'] = labels[:tr_size]
    data_dict['va']['x'] = features[tr_size:va_size+tr_size, :, :]
    data_dict['va']['t'] = labels[tr_size:va_size+tr_size]
    data_dict['ca']['x'] = features[va_size+tr_size:, :, :]
    # data_dict['ca']['t'] = labels[va_size+tr_size:]
    return data_dict


