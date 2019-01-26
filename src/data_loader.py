import numpy as np


path = '../datasets/'
sets = {'ag55': 'Ag_dataset.npz'}


def load(name):
    data = np.load(path + sets[name])
    labels = data['energies']
    features = data['Gs']
    features = (features - np.mean(features, axis=(0, 1))) / np.std(features, axis=(0, 1))
    # This is just used for testing the implementation
    #np.random.seed(1234)
    indices = np.random.permutation(len(features))

    tr_size = 5
    va_size = 1000
    te_size = 1000
    # all other samples are candidates

    data_dict = {'tr': dict(), 'va': dict(), 'ca': dict(), 'te': dict()}
    tr_range = indices[:tr_size]
    va_range = indices[tr_size:va_size+tr_size]
    te_range = indices[tr_size+va_size:tr_size+va_size+te_size]
    ca_range = indices[tr_size+va_size+te_size:]
    data_dict['tr']['x'] = features[tr_range]
    data_dict['tr']['t'] = labels[tr_range].astype(np.float64)
    data_dict['va']['x'] = features[va_range]
    data_dict['va']['t'] = labels[va_range].astype(np.float64)
    data_dict['ca']['x'] = features[ca_range]
    data_dict['ca']['t'] = labels[ca_range].astype(np.float64)
    data_dict['te']['x'] = features[te_range]
    data_dict['te']['t'] = labels[te_range].astype(np.float64)
    return data_dict


