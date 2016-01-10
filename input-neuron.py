# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

side_length = 125  # cm


def create_grid(side_length=side_length):
    input_x = np.linspace(-side_length/2., side_length/2., num=20).reshape(20,1)
    input_x = np.tile(input_x, (20, 1))
    
    input_y = np.linspace(-side_length/2., side_length/2., num=20).reshape(20,1)
    input_y = np.tile(input_y, 20)
    input_y = input_y.reshape(400, 1)
    
    input_neurons = np.concatenate((input_x, input_y), axis=1)
    return input_neurons


def input_rates(time_positions):
    input_neurons = create_grid()
    
    num_positions = time_positions.shape[0]
    num_inputs = input_neurons.shape[0]
    
    x_diff = input_neurons[:, 0].reshape(1,num_inputs) - time_positions[:,0].reshape(num_positions,1)
    y_diff = input_neurons[:, 1].reshape(1,num_inputs) - time_positions[:,1].reshape(num_positions,1)
    
    norm_sq = x_diff ** 2 + y_diff ** 2  
    
    rates = np.exp(- norm_sq / 50.)
    return rates


def test():
    #time_positions = np.random.rand(1000,3) * 125 - 62.5
    x = np.linspace(-60, 60, 1000).reshape(1000, 1)
    #x = np.linspace(0,0,1000).reshape(1000,1)
    y = np.linspace(-60, 60, 1000).reshape(1000, 1)
    #y = np.linspace(0,0,1000).reshape(1000,1)
    z = np.zeros((1000, 1))
    time_positions = np.concatenate([x, y, z], axis=1)
    rates = input_rates(time_positions)
    plt.pcolor(rates[780, :].reshape(20, 20))
    plt.colorbar()


def visualize():
    grid = create_grid()
    plt.figure(figsize=(10, 10))
    plt.scatter(grid[:,0], grid[:,1], marker='.')

    radius = np.sqrt(50.)    
    
    for i in range(grid.shape[0]):
        circle = plt.Circle((grid[i, 0], grid[i, 1], radius), fill=False, color='b')
        plt.gcf().gca().add_artist(circle)

    limit = 70.
    axes = plt.gca()
    axes.set_xlim([-limit, limit])
    axes.set_ylim([-limit, limit])
    
    axes.add_patch(patches.Rectangle((-62.5,-62.5), 125, 125, fill=False, color='r'))

visualize()

















