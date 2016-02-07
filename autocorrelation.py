# -*- coding: utf-8 -*-
import numpy as np

from scipy.signal import correlate2d

def autocorrelation(X):
    return correlate2d(X, X, boundary='fill', fillvalue=0.)

def autocorrelation_normalized(X):
    corr = correlate2d(X, X, boundary='fill', fillvalue=0.)
    one_corr = correlate2d(np.ones(np.shape(X)), np.ones(np.shape(X)), boundary='fill', fillvalue=0.)
    return np.divide(corr, one_corr)
