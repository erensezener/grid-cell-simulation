import numpy as np
import sklearn.preprocessing as pre


class FastOutputLayer:
    def __init__(self, input_layer_inputs):
        # layer metaparameters
        self.size = 40
        self.input_size = 400
        self.network_size = (self.size, self.input_size)
        self.a_bounds = (0.95, 1.05)
        self.a_0 = 1
        self.s_bounds = (2.95, 3.15)
        self.s_0 = 3
        self.input_layer_inputs = input_layer_inputs  # np matrix n by 400
        self.output_layer_outputs = np.zeros((np.size(input_layer_inputs, 0), self.size))  # time-step x 100
        self.recent_outputs = []  # for caching
        self.recent_outputs_computed = False  # for caching
        self.max_optim_iters = 40
        self.activation_fun = lambda p: 2 / np.pi * np.arctan(p[0] * (p[2] - p[1])) \
                                                    * (1 if p[2] - p[1] > 0 else 0) #p: [g, mu, r_plus]

        # layer parameters
        self.weights = np.random.uniform(0, 1, self.network_size)
        self.normalize_weights()
        self.avg_input_rates = np.zeros((self.input_size, ))
        self.avg_firing_rate = np.zeros((self.size, ))
        self.r_plus = 0.001 * np.ones((self.size, ))
        self.r_minus = 0.001 * np.ones((self.size, ))
        self.mu = 0 * np.ones((self.size, ))
        self.g = 4.5 * np.ones((self.size, ))
        self.s = 3 * np.ones((self.size, ))
        self.a = 1 * np.ones((self.size, ))
        self.b_mu = 0.01
        self.b_g = 0.1
        self.eta = 0.05
        self.epsilon = 0.005
        self.tau_plus = .1  # seconds
        self.tau_minus = .3  # seconds
        self.delta_t = 1e-3  # seconds


    # computes outputs for each time step from input layer outputs
    def process_all_inputs(self):
        for i in range(np.size(self.input_layer_inputs, 0)):
            print 'iteration: ' + str(i)
            input = self.input_layer_inputs[i, :]
            h = np.dot(self.weights, input)  # results in size: (100,)
            self.r_plus = self.r_plus + (1 / self.tau_plus) * (h - self.r_plus - self.r_minus) * self.delta_t
            self.r_minus = self.r_minus + (1 / self.tau_minus) * (h - self.r_minus) * self.delta_t
            # self.update_weights_fast(self.input_layer_inputs[i, :])
            for j in range(self.max_optim_iters):
                output = np.apply_along_axis(self.activation_fun,0,np.vstack((self.g, self.mu, self.r_plus)))
                a = np.mean(output)
                s = np.std(output)

                self.mu += self.b_mu * (a - self.a_0)
                self.g += self.b_g * self.g * (s - self.s_0)

                if (self.a_bounds[0] < a < self.a_bounds[1]) \
                    and (self.s_bounds[0] < s < self.s_bounds[1]):
                    self.a = a
                    self.s = s
                    break
            #if no satisfactory a's are found, use the last values
            self.a = a
            self.s = s

            self.avg_firing_rate += self.eta * (output - self.avg_firing_rate) #size: 100
            self.avg_input_rates += self.eta * (input - self.avg_input_rates) #size: 400
            self.weights += self.epsilon * (np.outer(output, input) - np.outer(self.avg_firing_rate, self.avg_input_rates))
            self.normalize_weights()


    def normalize_weights(self):
        normalized_weights = pre.normalize(self.weights, norm='l2', copy='true')
        # sanity check start
        column_sum = np.sum(np.power(normalized_weights, 2), axis=1)
        assert 0.999 < column_sum[0] < 1.001  # sanity check
        # sanity check end
        self.weights = normalized_weights



def main():
    layer = FastOutputLayer(np.zeros((100, 400)))
    layer.process_all_inputs()


if __name__ == '__main__':
    main()
