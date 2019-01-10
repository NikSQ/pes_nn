import numpy as np
import matplotlib.pyplot as plt

n_exp = 2
legend = ['highest variance', 'random']
suffix = 'std'
epochs = np.load('numerical_results/' + suffix + '_0_v_epochs.npy')

def plot(key):
  plt.figure()
  for exp_idx in range(n_exp):           
    partial_name = suffix + '_' + str(exp_idx)
    data = np.load('numerical_results/' + partial_name + '_' + key + '.npy')
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)
    plt.errorbar(x=epochs, y=mean, yerr=std, fmt='o', capsize=5)

  plt.xlabel('epoch')
  plt.ylabel('output variance')
  plt.legend(legend, loc='upper right')
  plt.savefig('plots/var/' + key + '.png')
  

plot('v_ordered')
plot('v_random')
 
    

  
