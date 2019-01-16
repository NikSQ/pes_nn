import numpy as np


path = '../datasets/'
sets = {'ag55': 'Ag_dataset.npz'}


def load(name):
    data = np.load(path + sets[name])
    labels = data['energies']
    features = data['Gs']
    features = (features - np.mean(features, axis=(0, 1))) / np.std(features, axis=(0, 1))
    # This is just used for testing the implementation
    np.random.seed(1234)
    indices = np.random.permutation(len(features))

    tr_size = 5000
    va_size = 2500
    # all other samples are candidates

    data_dict = {'tr': dict(), 'va': dict(), 'ca': dict()}
    data_dict['tr']['x'] = features[indices[:tr_size]]
    data_dict['tr']['t'] = labels[indices[:tr_size]].astype(np.float64)
    data_dict['va']['x'] = features[indices[tr_size:va_size+tr_size]]
    data_dict['va']['t'] = labels[indices[tr_size:va_size+tr_size]].astype(np.float64)
    data_dict['ca']['x'] = features[indices[va_size+tr_size:]]
    data_dict['ca']['t'] = labels[indices[va_size+tr_size:]].astype(np.float64)
    return data_dict


