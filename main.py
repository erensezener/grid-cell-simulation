import output_layer
import numpy as np

o = output_layer.OutputLayer(np.zeros((10,400)))
o.process_all_inputs()
print o.output_layer_outputs
# print sum(o.get_weights(1))