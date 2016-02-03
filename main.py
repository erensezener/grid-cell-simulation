import fast_output_layer
import input_neuron
import rat_simulator
import matplotlib.pyplot as plt
import datetime
import numpy as np
from scipy import sparse, io
import autocorrelation
import os

SUB_DIR = '3.6M'
FILENAME = 'data/Rat_Data_36M.txt'
INPUT_LAYER_OUTPUTS = 'results/' + SUB_DIR + '/input_layer.mtx'
# FILENAME = 'data/Rat_Data.txt'

if not os.path.isdir(SUB_DIR):
    os.makedirs(SUB_DIR)


print 'reading data'
rat_data = rat_simulator.rat_txt_to_matrix(FILENAME)
print 'data read'
rat_data = rat_data[:1e6,:]
print 'number of data points: ' + str(np.shape(rat_data))

a = datetime.datetime.now().replace(microsecond=0)
input_layer_outputs = input_neuron.input_rates_sparse(rat_data)
b = datetime.datetime.now().replace(microsecond=0)
io.mmwrite(INPUT_LAYER_OUTPUTS, input_layer_outputs) #write sparse matrix to a file in COO format
print 'number of nnz from input layer ' + str(input_layer_outputs.nnz)
print 'input processed in ' + str(b-a)

a = datetime.datetime.now().replace(microsecond=0)
output_layer = fast_output_layer.FastOutputLayer(input_layer_outputs)
output_layer.process_all_inputs()
b = datetime.datetime.now().replace(microsecond=0)
print 'output processed in ' + str(b-a)

for i in range(100):
    weights = input_neuron.reshape_vec_to_grid(output_layer.weights[i, :])
    plt.matshow(weights)
    plt.colorbar()
    plt.title('Weights of neuron ' + str(i))
    plt.savefig('results/' + SUB_DIR + '/neuron_w_' + str(i) +'.png', format="png")
    plt.close()

    plt.matshow(autocorrelation.autocorrelation(weights))
    plt.colorbar()
    plt.title('Autocorr. of neuron ' + str(i))
    plt.savefig('results/' + SUB_DIR + '/neuron_a_' + str(i) +'.png', format="png")
    plt.close()



# plt.plot(output_layer.output_layer_outputs)
# print output_layer_outputs.output_layer_outputs[998,:]
# plt.show()
# plt.matshow(output_layer.weights)
# plt.colorbar()
# plt.show()

# plt.plot(output_layer.g_history)
# plt.title('g history')
# plt.savefig('figures/g_history.png', format="png")
# plt.close()
#
# plt.plot(output_layer.mu_history)
# plt.title('mu history')
# plt.savefig('figures/mu_history.png', format="png")
# plt.close()
#
# plt.plot(output_layer.iter_no_history)
# plt.title('g history')
# plt.savefig('figures/iter_history.png', format="png")
# plt.close()
