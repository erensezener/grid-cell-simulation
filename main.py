import output_layer
import input_neuron
import numpy as np
import MNS_Project_New
import matplotlib.pyplot as plt

FILENAME = 'Rat_Data.txt'

print 'reading data'
rat_data = MNS_Project_New.rat_txt_to_matrix(FILENAME)
print 'data read'
rat_data = rat_data[1:10,:]
input_layer_outputs = input_neuron.input_rates(rat_data)
print 'input processed'
output_layer_outputs = output_layer.OutputLayer(input_layer_outputs)
output_layer_outputs.process_all_inputs()
print 'output processed'

print output_layer_outputs.output_layer_outputs
plt.plot(output_layer_outputs.output_layer_outputs)
plt.bar(output_layer_outputs.output_layer_outputs[8])
# print output_layer_outputs.output_layer_outputs[9]
# print sum(o.get_weights(1))