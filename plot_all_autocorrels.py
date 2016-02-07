import numpy as np
import autocorrelation
import matplotlib.pyplot as plt
import input_neuron

SUB_DIR = './results/9M'
FILE = 'weights.npz'
f = np.load(SUB_DIR + '/' + FILE)

network_size = (100, 400)

arr = f['arr_0']
(no_rows, no_cols) = np.shape(arr)
last_weights = np.reshape(arr[no_rows-1,:], (network_size[0], 400), order='F')

for i in range(network_size[0]):
    weights = input_neuron.reshape_vec_to_grid(last_weights[i,:])
    plt.matshow(weights)
    plt.colorbar()
    plt.title('Weights of neuron ' + str(i))
    plt.savefig(SUB_DIR + '/neuron_w_' + str(i) +'.png', format="png")
    plt.close()

    plt.matshow(autocorrelation.autocorrelation_normalized(weights))
    plt.colorbar()
    plt.title('Autocorr. of neuron ' + str(i))
    plt.savefig(SUB_DIR + '/neuron_a_' + str(i) +'.png', format="png")
    plt.close()

