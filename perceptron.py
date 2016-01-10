import numpy as np
import sklearn.preprocessing as pre


class Perceptron:
    def __init__(self):
        self.activation_fun = lambda g, mu, r_plus: 2 / np.pi * np.arctan(g * (r_plus - mu)) \
                                                    * (1 if r_plus - mu > 0 else 0)
        self.weights = np.random.uniform(0, 1, 400)
        self.normalize_weights()
        self.r_plus = 0  # TODO initial value?
        self.r_minus = 0  # TODO initial value?
        self.mu = 0  # TODO initial value?
        self.a = 0.1
        self.s = 0.3


    def compute_h(self, r):
        if len(np.shape(r)) is not 1:
            r = np.transpose(r)
            return np.sum(np.dot(np.transpose(self.weights), r))
        else:
            return np.dot(np.transpose(self.weights), r)

    def compute_output(self, x):
        return self.activation_fun(self.compute_h(x))

    def update(self, delta_w):
        self.weights += np.transpose(delta_w)

    def normalize_weights(self):
        normalized_weights = pre.normalize(self.weights, norm='l2', copy='true')
        assert 0.999 < np.sum([w**2 for w in normalized_weights.reshape((400,))]) < 1.001 # sanity check
        self.weights = normalized_weights.reshape((400,))

    def get_weights(self):
        return self.weights

    def get_weight(self, i):
        return self.weights[i]
