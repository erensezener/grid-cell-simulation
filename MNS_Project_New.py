
# coding: utf-8

# In[30]:

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
        while abs(x)>=x_lim or abs(y)>=y_lim:
            phi=phi+(np.pi*0.5)
            x,y=compute_new_position(x,y,phi,r)     
        t=t+timestep       
        output[i+1,0]=x
        output[i+1,1]=y
        output[i+1,2]=t
    return output

def plot_rat_trajectory(rat_matrix):
    plt.plot(rat_matrix[:,0], rat_matrix[:,1])
    plt.xlim(-65,65)
    plt.ylim(-65,65)
    plt.show()

def rat2txt(rat_matrix):
    data= open('Rat_Data.txt', 'w')
    for i in range(len(rat_matrix[:,1])):
        data.write(str(rat_matrix[i,0])+ "\t" +str(rat_matrix[i,1])+ "\t" +str(rat_matrix[i,2])+"\n")
    data.close()
    
def rat_txt_to_matrix(filename):
    data=open(str(filename), 'r')
    outputmatrix=np.zeros(3)
    for line in data:
        values=line.split()
        v=np.zeros(3)
        v[0]=float(values[0])
        v[1]=float(values[1])
        v[2]=float(values[2])
        outputmatrix= np.vstack((outputmatrix,v))
    return outputmatrix

rat=running_rat(4000000)
plot_rat_trajectory(rat)
rat2txt(rat)
#print rat_txt_to_matrix("Rat_Data.txt")


# In[ ]:




# In[ ]:



