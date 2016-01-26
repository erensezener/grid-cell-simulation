import fast_output_layer
import input_neuron
import MNS_Project_New
import matplotlib.pyplot as plt
import datetime

FILENAME = 'Rat_Data.txt'

print 'reading data'
rat_data = MNS_Project_New.rat_txt_to_matrix(FILENAME)
print 'data read'
rat_data = rat_data[1:100,:]
input_layer_outputs = input_neuron.input_rates(rat_data)
print 'input processed'
a = datetime.datetime.now().replace(microsecond=0)
output_layer_outputs = fast_output_layer.FastOutputLayer(input_layer_outputs)
output_layer_outputs.process_all_inputs()
b = datetime.datetime.now().replace(microsecond=0)
print 'output processed in ' + str(b-a)

print output_layer_outputs.output_layer_outputs
plt.plot(output_layer_outputs.output_layer_outputs)
plt.show()
# plt.bar(output_layer_outputs.output_layer_outputs[8])
# print output_layer_outputs.output_layer_outputs[9]
# print sum(o.get_weights(1))