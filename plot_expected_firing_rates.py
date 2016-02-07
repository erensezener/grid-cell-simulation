import numpy as np
import autocorrelation
import matplotlib.pyplot as plt
import input_neuron

neurons_indices_to_plot = [12, 14, 16, 41]

PATH = './results/9M/'
WEIGHTS_FILE = 'weights.npz'
weight_f = np.load(PATH + WEIGHTS_FILE)


network_size = (100, 400)

arr = weight_f['arr_0']
(no_rows, no_cols) = np.shape(arr)

firing_rate = np.zeros(50, 50)

last_weights = np.reshape(arr[no_rows - 1, :], (network_size[0], 400), order='F')

#generate rat data
for i in np.linspace(-62.5, 62.5, 50):
    for j in np.linspace(-62.5, 62.5, 50):
        r = input_neuron.input_rates(np.array([i, j]))
        h = np.dot(last_weights, r)
        firing_rate[i,j] = activation_fun()


for time_step in np.linspace(0, 100, 11):

    last_weights = np.reshape(arr[time_step, :], (network_size[0], 400), order='F')

    for i in neurons_indices_to_plot:
        weights = input_neuron.reshape_vec_to_grid(last_weights[i, :])

        # plt.matshow(weights)
        # plt.colorbar()
        # plt.title('Weights of neuron ' + str(i) + 'at time step' + str(time_step))
        # plt.savefig(PATH + '/selected_neurons/2neuron_w_' + str(i) + '_t_' + str(time_step) + '.png', format="png")
        # plt.close()

        plt.matshow(autocorrelation.autocorrelation_normalized(weights))
        plt.colorbar()
        plt.title('Autocorr. of neuron ' + str(i) + 'at time step' + str(int(time_step)))
        plt.savefig(PATH + '/selected_neurons/2neuron_a_' + str(i) + '_t_' + str(int(time_step)) + '.png', format="png")
        plt.close()

def activation_fun(g, mu, r_plus):
    val = 2.0 / np.pi * np.arctan(g * (r_plus - mu)) * 0.5 * (np.sign(r_plus - mu) + 1.0)
    return val