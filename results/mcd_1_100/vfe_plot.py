import numpy as np
import matplotlib.pyplot as plt

n_exp = 2
legend = ['Highest variance', 'Random']
suffix = 'std'
epochs = np.load('numerical_results/' + suffix + '_0_epochs.npy')

def plot(key):
  plt.figure()
  for exp_idx in range(n_exp):           
    partial_name = suffix + '_' + str(exp_idx)
    data = np.squeeze(np.load('numerical_results/' + partial_name + '_' + key + '.npy'))
    plt.plot(epochs, data, '-')

  plt.xlabel('epoch')
  plt.ylabel('MSE')
  plt.xlim([0, 5000])
  plt.ylim([0, 1.])
  plt.legend(legend, loc='upper right')
  plt.savefig('plots/vfe/' + key + '.png')
  

plot('tr_vfe')
plot('va_vfe')
 
    

  
