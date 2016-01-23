# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint

tau_plus = .1       # seconds
tau_minus = .3      # seconds

# oscillation frequency of h
k = .1              # 1/seconds^2

def h(t):
    return np.sin(np.pi * k * t ** 2)

def f(r, t):
    f1 = (1 / tau_plus) * (h(t) - r[0] - r[1])
    f2 = (1 / tau_minus) * (h(t) - r[1])
    return np.array([f1, f2])
    
# which initial values for r_plus and r_minus?

r_0 = np.random.uniform(.1,.9,size=2) # r_0 = [r_plus_0, r_minus_0]

def adaptation(times):
    # assuming equidistant time points
    refinement_factor = 10
    times = times * 1000 # moving from milliseconds to seconds
    fine_times = np.linspace(times[0], times[-1], refinement_factor * (len(times)-1) + len(times))
    r = odeint(f, r_0, fine_times) # r = [r_plus, r_minus]
    return r[::refinement_factor + 1] #subsampling
