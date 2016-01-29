import numpy as np
import perceptron


class OutputLayer:
    def __init__(self, input_layer_inputs):
        self.size = 100
        self.a = 1
        self.a_bounds = (0.95, 1.05)
        self.s = 3
        self.s_bounds = (2.95, 3.15)
        self.output_layer = [perceptron.Perceptron(self.a, self.a)
                             for i in range(self.size)]
        self.input_layer_inputs = input_layer_inputs  # np matrix n by 400
        self.output_layer_outputs = np.zeros((np.size(input_layer_inputs, 0), self.size))
        self.recent_outputs = []
        self.recent_outputs_computed = False
        self.max_optim_iters = 10

    # computes outputs for each time step from input layer outputs
    def process_all_inputs(self):
        for i in range(np.size(self.input_layer_inputs, 0)):
            print 'iteration: ' + str(i)
            self.update_weights(self.input_layer_inputs[i, :])
            self.update_a_and_s()
            self.update_rs(self.input_layer_inputs[i, :])
            outputs = self.compute_outputs()
            self.output_layer_outputs[i, :] = np.reshape(outputs, (1, self.size))

    # computes outputs (caches the most recent output)
    def compute_outputs(self):
        if not self.recent_outputs_computed:
            self.recent_outputs = [p.compute_output() for p in self.output_layer]
            self.recent_outputs_computed = True
        assert len(self.recent_outputs) is not 0
        return self.recent_outputs


    def update_a_and_s(self):
        a, s = 0, 0  # temp values
        for i in range(self.max_optim_iters):
            a = self.compute_a()
            s = self.compute_s()
            self.update_mu_g(s, a)
            if (self.a_bounds[0] < a < self.a_bounds[1]) \
                    and (self.s_bounds[0] < s < self.s_bounds[1]):
                self.a = a
                self.s = s
                break

        # if self.max_optim_iters many iters is not enough
        self.a = a
        self.s = s

    def update_mu_g(self, a, s):
        map(lambda n: n.update_mu(a), self.output_layer)
        map(lambda n: n.update_g(s), self.output_layer)
        self.recent_outputs_computed = False


    def compute_a(self):
        return sum(self.compute_outputs()) / self.size

    def compute_s(self):
        return sum(self.compute_outputs()) ** 2 / \
               (self.size * sum(map(lambda x: x**2, self.compute_outputs())))

    def get_weights(self, i):
        return self.output_layer[i].get_weights()

    def update_rs(self, r):
        map(lambda o: o.update_r_plus(r), self.output_layer)

    # normalization (i.e., fixed L2 norm) is done in Perceptron.update()
    def update_weights(self, r):
        map(lambda o: o.update(r), self.output_layer)
        self.recent_outputs_computed = False
