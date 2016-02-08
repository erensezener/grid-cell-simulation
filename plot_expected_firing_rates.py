import numpy as np
import autocorrelation
import matplotlib.pyplot as plt
import input_neuron

neurons_indices_to_plot = range(100)

PATH = './results/9M_2/'
WEIGHTS_FILE = 'weights.npz'
PARAMS_FILE = 'params.npz'
params_f = np.load(PATH + PARAMS_FILE)
weight_f = np.load(PATH + WEIGHTS_FILE)

network_size = (100, 400)

w_arr = weight_f['arr_0']
(no_rows, no_cols) = np.shape(w_arr)

p_arr = params_f['arr_0']

firing_rate = np.zeros((50, 50))
g = p_arr[-1,0]
mu = p_arr[-1,1]

last_weights = np.reshape(w_arr[- 1, :], (network_size[0], 400), order='F')


def activation_fun(g, mu, r_plus):
    val = 2.0 / np.pi * np.arctan(g * (r_plus - mu)) * 0.5 * (np.sign(r_plus - mu) + 1.0)
    return val

#generate rat data
j_vals = i_vals = np.linspace(-62.5, 62.5, 50)
for k in range(100):
    for i in range(50):
        for j in range(50):
            i_val = i_vals[i]
            j_val = j_vals[j]
            r = input_neuron.input_rates(np.array([[i_val, j_val]]))
            h = np.dot(last_weights[k,:], r.transpose())
            firing_rate[i,j] = activation_fun(g, mu, h)

    plt.matshow(np.rot90(firing_rate))
    plt.colorbar()
    plt.title('The final firing rate of neuron ' + str(k))
    plt.axes([-62.5, 62.5, -62.5, 62.5])
    # plt.xlabel(np.linspace(-62.5, 62.5, 10))
    # plt.yticks(np.linspace(-62.5, 62.5, 10))
    plt.savefig(PATH + 'neuron_r_' + str(k) + '.png', format="png")
    plt.close()
    # plt.show()

# for time_step in np.linspace(0, 100, 11):
#
#     last_weights = np.reshape(w_arr[time_step, :], (network_size[0], 400), order='F')
#
#     for i in neurons_indices_to_plot:
#         weights = input_neuron.reshape_vec_to_grid(last_weights[i, :])
#
#         # plt.matshow(weights)
#         # plt.colorbar()
#         # plt.title('Weights of neuron ' + str(i) + 'at time step' + str(time_step))
#         # plt.savefig(PATH + '/selected_neurons/2neuron_w_' + str(i) + '_t_' + str(time_step) + '.png', format="png")
#         # plt.close()
#
#         plt.matshow(autocorrelation.autocorrelation_normalized(weights))
#         plt.colorbar()
#         plt.title('Autocorr. of neuron ' + str(i) + 'at time step' + str(int(time_step)))
#         plt.savefig(PATH + '/selected_neurons/2neuron_a_' + str(i) + '_t_' + str(int(time_step)) + '.png', format="png")
#         plt.close()
