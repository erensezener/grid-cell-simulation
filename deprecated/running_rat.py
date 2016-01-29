# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def compute_new_position(x,y,phi,r):
    x_new=x+(r*np.cos(phi))
    y_new=y+(r*np.sin(phi))
    return x_new,y_new
    

def running_rat(t_max, timestep=10.0, velocity=40, x_lim=62.5, y_lim=62.5): #length&width in cm; velocity in cm/s; t_max in ms
    print 'calculating ratjectory...'
    r=velocity*0.01
    t=0
    output= np.zeros((int(t_max/timestep), 3))
    
    x=62.5-125*(np.random.random_sample())
    y=62.5-125*(np.random.random_sample())
    output[0,0]=x
    output[0,1]=y
    phi=np.arctan(y/x)
    for i in range(int(t_max/timestep)-1):
        phi=np.random.normal(loc=phi, scale=0.2)
        x,y=compute_new_position(x,y,phi,r)           
        while abs(x)>=62.5 or abs(y)>=62.5:
            phi= np.random.normal(loc=phi, scale=1.0)
            x,y=compute_new_position(x,y,phi,r)
        t=t+timestep
        output[i+1,0]=x
        output[i+1,1]=y
        output[i+1,2]=t
    return output
