import fast_output_layer
import input_neuron
import rat_simulator
import matplotlib.pyplot as plt
import datetime
import numpy as np


FILENAME = 'data/Rat_Data_36M.txt'
# FILENAME = 'Rat_Data_short.txt'


print 'reading data'
rat_data = rat_simulator.rat_txt_to_matrix(FILENAME)
print 'data read'
rat_data = rat_data[1:1e4,:]
print 'number of data points: ' + str(np.shape(rat_data))

a = datetime.datetime.now().replace(microsecond=0)
input_layer_outputs = input_neuron.input_rates(rat_data)
b = datetime.datetime.now().replace(microsecond=0)
print 'input processed in ' + str(b-a)

a = datetime.datetime.now().replace(microsecond=0)
output_layer = fast_output_layer.FastOutputLayer(input_layer_outputs)
output_layer.process_all_inputs()
b = datetime.datetime.now().replace(microsecond=0)
print 'output processed in ' + str(b-a)

for i in range(40):
    plt.matshow(input_neuron.reshape_vec_to_grid(output_layer.weights[i, :]))
    plt.colorbar()
    plt.title('Neuron ' + str(i))
    plt.savefig('figures/neuron_' + str(i) +'.png', format="png")
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
