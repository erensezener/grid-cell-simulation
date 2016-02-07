import fast_output_layer_batch
import input_neuron
import rat_simulator
import matplotlib.pyplot as plt
import datetime
import numpy as np
import autocorrelation


SUB_DIR = '36M_2'
FILENAME = 'data/Rat_Data_9M.txt'
# INPUT_LAYER_OUTPUTS = 'results/' + SUB_DIR + '/input_layer.mtx'

NUMBER_OF_BATCHES = 100

print 'reading data'
rat_data = rat_simulator.rat_txt_to_matrix(FILENAME)
print 'data read'
# rat_data = rat_data[:1e4,:]
print 'number of data points: ' + str(np.shape(rat_data))
number_of_data_points = np.size(rat_data, 0)

output_layer = fast_output_layer_batch.FastOutputLayerBatch(number_of_data_points, NUMBER_OF_BATCHES)

batch_size = int(number_of_data_points / NUMBER_OF_BATCHES)

for i in range(NUMBER_OF_BATCHES):
    print 'batch: ' + str(i)
    a = datetime.datetime.now().replace(microsecond=0)
    input_layer_outputs = input_neuron.input_rates_sparse(rat_data[i*batch_size:((i+1)*batch_size), :])
    b = datetime.datetime.now().replace(microsecond=0)
    # io.mmwrite(INPUT_LAYER_OUTPUTS, input_layer_outputs) #write sparse matrix to a file in COO format
    print 'number of nnz from input layer ' + str(input_layer_outputs.nnz)
    print 'input processed in ' + str(b-a)

    a = datetime.datetime.now().replace(microsecond=0)
    output_layer.process_all_inputs(input_layer_outputs)
    b = datetime.datetime.now().replace(microsecond=0)
    print 'output processed in ' + str(b-a)

output_layer.save_data_to_disk()


for i in range(100):
    weights = input_neuron.reshape_vec_to_grid(output_layer.weights[i, :])
    plt.matshow(weights)
    plt.colorbar()
    plt.title('Weights of neuron ' + str(i))
    plt.savefig('results/' + SUB_DIR + '/neuron_w_' + str(i) +'.png', format="png")
    plt.close()

    plt.matshow(autocorrelation.autocorrelation_normalized(weights))
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
