import numpy as np


path = '../datasets/'
sets = {'ag55': 'Ag_dataset.npz'}


def load(name, tr_idc=None):
    data = np.load(path + sets[name])
    labels = data['energies']
    features = data['Gs']
    features = (features - np.mean(features, axis=(0, 1))) / np.std(features, axis=(0, 1))

    tr_size = 5
    va_size = 1000
    te_size = 1000

    indices = np.random.permutation(len(features))
    if tr_idc is not None:
        tr_range = tr_idc
        indices = np.delete(indices, np.where(np.isin(indices, tr_range)))
        va_range = indices[:1000]
        te_range = indices[1000:]
        print(len(te_range))
        data_dict = {'tr': dict(), 'va': dict(), 'te': dict()}
        data_dict['tr']['x'] = features[tr_range]
        data_dict['tr']['t'] = labels[tr_range].astype(np.float64)
        data_dict['va']['x'] = features[va_range]
        data_dict['va']['t'] = labels[va_range].astype(np.float64)
        data_dict['te']['x'] = features[te_range]
        data_dict['te']['t'] = labels[te_range].astype(np.float64)
        data_dict['tr']['range'] = np.asarray(tr_range)
    else:
        tr_range = indices[:tr_size]
        indices = np.delete(indices, np.where(np.isin(indices, tr_idc)))

        # all other samples are candidates
        va_range = indices[:va_size]
        te_range = indices[va_size:va_size+te_size]
        ca_range = indices[va_size+te_size:]

        data_dict = {'tr': dict(), 'va': dict(), 'ca': dict(), 'te': dict()}
        data_dict['tr']['x'] = features[tr_range]
        data_dict['tr']['t'] = labels[tr_range].astype(np.float64)
        data_dict['va']['x'] = features[va_range]
        data_dict['va']['t'] = labels[va_range].astype(np.float64)
        data_dict['ca']['x'] = features[ca_range]
        data_dict['ca']['t'] = labels[ca_range].astype(np.float64)
        data_dict['te']['x'] = features[te_range]
        data_dict['te']['t'] = labels[te_range].astype(np.float64)
        data_dict['ca']['range'] = np.asarray(ca_range)
        data_dict['tr']['range'] = np.asarray(tr_range)
    return data_dict


