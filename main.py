import output_layer
import input_neuron
import numpy as np


input_layer_outputs = input_neuron.input_rates(np.zeros((10,400)))
assert np.shape(input_layer_outputs) == (10,400)
print 'Input processed'
output_layer_outputs = output_layer.OutputLayer(input_layer_outputs)
output_layer_outputs.process_all_inputs()
print 'Output processed'

print output_layer_outputs.output_layer_outputs
# print sum(o.get_weights(1))