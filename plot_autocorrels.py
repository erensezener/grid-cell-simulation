import numpy as np
import autocorrelation
import matplotlib.pyplot as plt
import input_neuron

PATH = 'results/3.6M/'
FILE = 'weights.npz'
f = np.load(PATH + FILE)

network_size = (40, 400)

arr = f['arr_0']
(no_rows, no_cols) = np.shape(arr)
last_weights = np.reshape(arr[no_rows-2,:], (40, 400), order='F')

for i in range(network_size[0]):
    weights = input_neuron.reshape_vec_to_grid(last_weights[i,:])
    plt.matshow(weights)
    plt.colorbar()
    plt.title('Neuron ' + str(i))
    plt.show()
    plt.matshow(autocorrelation.autocorrelation(weights))
    plt.show()
