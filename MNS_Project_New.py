
# coding: utf-8

# In[ ]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt


def compute_new_position(x,y,phi,r):
    x_new=x+(r*np.cos(phi))
    y_new=y+(r*np.sin(phi))
    return x_new,y_new
    

def running_rat(t_max, timestep=10.0, velocity=40, x_lim=62.5, y_lim=62.5): #length&width in cm; velocity in cm/s; t_max in ms
    r=velocity*0.01
    t=0
    output= np.zeros((int(t_max/timestep), 3))
    turn=[-np.pi*0.5, np.pi*0.5]
    
    x=62.5-125*(np.random.random_sample())
    y=62.5-125*(np.random.random_sample())
    output[0,0]=x
    output[0,1]=y
    phi=np.arctan(y/x)
    for i in range(int(t_max/timestep)-1):
        phi=np.random.normal(loc=phi, scale=0.2)
        x,y=compute_new_position(x,y,phi,r)
        k=0
        while abs(x)>=x_lim or abs(y)>=y_lim:
            if k<10:
                turn_arg=np.random.randint(0, high=2)
                phi=phi+turn[turn_arg]                
            else:
                phi=phi+(np.pi*0.5)
            x,y=compute_new_position(x,y,phi,r)     
            k=k+1
        t=t+timestep       
        output[i+1,0]=x
        output[i+1,1]=y
        output[i+1,2]=t
    return output

rat=running_rat(10000)
plt.plot(rat[:,0], rat[:,1])
plt.xlim(-65,65)
plt.ylim(-65,65)
plt.show()


# In[ ]:




# In[ ]:



