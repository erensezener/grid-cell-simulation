import numpy as np
import autocorrelation
import matplotlib.pyplot as plt
import input_neuron

neurons_indices_to_plot = [20, 31]

PATH = './results/9M_2/'
WEIGHTS_FILE = 'weights.npz'
weight_f = np.load(PATH + WEIGHTS_FILE)

network_size = (100, 400)

arr = weight_f['arr_0']
(no_rows, no_cols) = np.shape(arr)

for time_step in np.linspace(0, 100, 11):

    last_weights = np.reshape(arr[time_step, :], (network_size[0], 400), order='F')

    for i in neurons_indices_to_plot:
        weights = input_neuron.reshape_vec_to_grid(last_weights[i, :])

        plt.matshow(weights)
        plt.colorbar()
        plt.title('Weights of neuron ' + str(i) + 'at time step' + str(time_step))
        plt.savefig(PATH + '/selected_neurons/neuron_w_' + str(i) + '_t_' + str(int(time_step)) + '.png', format="png")
        plt.close()

        plt.matshow(autocorrelation.autocorrelation_normalized(weights))
        plt.colorbar()
        plt.title('Autocorr. of neuron ' + str(i) + 'at time step' + str(int(time_step)))
        plt.savefig(PATH + '/selected_neurons/neuron_a_' + str(i) + '_t_' + str(int(time_step)) + '.png', format="png")
        plt.close()
