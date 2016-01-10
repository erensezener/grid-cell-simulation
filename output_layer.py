import numpy as np
import perceptron

class OutputLayer:
    def __init__(self):
        self.output_layer = [perceptron.Perceptron() for i in range(100)]

    def compute_outputs(self, x):
        return [p.compute_output(x) for p in self.output_layer]

    def update(self):
        pass

    def get_weights(self, i):
        return self.output_layer[i].get_weights()
