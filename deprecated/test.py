# -*- coding: utf-8 -*-

import input_neuron
import running_rat

print('calling running_rat')
positions = running_rat.running_rat(1e4)

print('calling input_rates')
rates = input_neuron.input_rates(positions)

input_neuron.visualize_activity(positions, rates)
