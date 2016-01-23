import numpy as np
import sklearn.preprocessing as pre


class Perceptron:
    def __init__(self, a_0, s_0):
        self.activation_fun = lambda g, mu, r_plus: 2 / np.pi * np.arctan(g * (r_plus - mu)) \
                                                    * (1 if r_plus - mu > 0 else 0)
        self.input_size = 400
        self.weights = np.random.uniform(0, 1, self.input_size)
        self.normalize_weights()
        self.avg_input_rates = np.zeros((1, self.input_size))
        self.avg_firing_rate = 0
        self.r_plus = 1  # TODO initial value?
        self.r_minus = 1  # TODO initial value?
        self.mu = 0
        self.g = 4.5
        self.a_0 = a_0
        self.s_0 = s_0
        self.b_mu = 0.01
        self.b_g = 0.1
        self.eta = 0.05
        self.epsilon = 0.005
        self.tau_plus = .1 # seconds
        self.tau_minus = .3 # seconds
        self.delta_t = 1e-3 #seconds

    def update_mu(self, a):
        self.mu += self.b_mu * (a - self.a_0)

    def update_g(self, s):
        self.g += self.b_g * self.g * (s - self.s_0)

    def update_r_plus(self, r):
        h = self.compute_h(r)
        self.r_plus = self.r_plus + (1 / self.tau_plus) * (h - self.r_plus - self.r_minus) * self.delta_t
        self.r_minus = self.r_minus + (1 / self.tau_minus) * (h - self.r_minus) * self.delta_t

    def compute_h(self, r):
        if len(np.shape(r)) is not 1:
            r = np.transpose(r)
            return np.sum(np.dot(np.transpose(self.weights), r))
        else:
            return np.dot(np.transpose(self.weights), r)

    def compute_output(self):
        return self.activation_fun(self.g, self.mu, self.r_plus)

    def update(self, r):
        o = self.compute_output()
        self.avg_firing_rate += self.eta * (o - self.avg_firing_rate)
        for i in range(self.input_size):
            self.avg_input_rates[:,i] += self.eta * (r[i] - self.avg_input_rates[:,i])
            self.weights[i] += self.epsilon * (o * r[i] - self.avg_firing_rate * self.avg_input_rates[:,i])

        self.normalize_weights()

    def normalize_weights(self):
        normalized_weights = pre.normalize(self.weights, norm='l2', copy='true')
        assert 0.999 < np.sum([w ** 2 for w in normalized_weights.reshape((self.input_size,))]) < 1.001  # sanity check
        self.weights = normalized_weights.reshape((self.input_size,))

    def get_weights(self):
        return self.weights

    def get_weight(self, i):
        return self.weights[i]
