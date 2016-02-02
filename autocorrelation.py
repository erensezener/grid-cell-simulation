# -*- coding: utf-8 -*-

from scipy.signal import correlate2d

def autocorrelation(X):
    return correlate2d(X, X, boundary='fill', fillvalue=0.)
