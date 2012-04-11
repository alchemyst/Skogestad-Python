import numpy as np
import scipy as sc
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from mpl_toolkits.mplot3d import Axes3D
from Perturbations import Possibilities 

#drawing circles in all directions with G and Gd to see were problem could be expected and in what direction


def Gd(s):
    """disturbance transfer function matrix 
    needs to have the same amount of columns than G"""
    Gd= np.matrix([[1/(s+1), s**2/((s+2)*(s+4)), (s+1)/((s+8)*(s+4)*(s+2))], [1/(s+10), 4, s*(s+1)/((s+5)*(s+3))]])
    return Gd


def G(s):
    """system transfer function matrix
    needs to have the same amount of columns than Gd"""
    G =np.matrix([[1/(s+0.1), (s+1)/((s+3)*(s+10))], [1/(s+100), (s+1)*(s+2)/((s+3)*(s+30))]])
    return G

def directions(Amount,w):
    """give all possible directions of a certain transfer function matrix
    to be used with SVD to plot the magnitude and direction in all directions
    This would give indication where control problems would occurs"""

    #generating all possible input directions for Gd
    umin_Gd = np.matrix(np.zeros([Gd(0.001).shape[1], 1]))
    umax_Gd = np.matrix(np.ones([Gd(0.001).shape[1], 1]))
    Possibilities_plant_Gd = Possibilities(umin_Gd, umax_Gd, Amount)
    
    #generating all possible input direction for G
    umin_G = np.matrix(np.zeros([G(0.001).shape[1], 1]))
    umax_G = np.matrix(np.ones([G(0.001).shape[1], 1]))
    Possibilities_plant_G = Possibilities(umin_G, umax_G, Amount)

    #generating outputs from the input direction unit step changes
    Output_direction_Gd_non_normalized = [Gd(1j*w)*Possibilities_plant_Gd[i, :].T for i in range(Possibilities_plant_Gd.shape[0])]
    Output_direction_G_non_normalized =[G(1j*w)*Possibilities_plant_G[i, :].T for i in range(Possibilities_plant_G.shape[0])]
    
    Output_direction_Gd_normalized
    Output_direction_G_normalized   
    

directions(10, 0.001)

umin=np.zeros([3, 1])
umax=np.ones([3, 1])

print Possibilities(umin, umax, 3)[0,:]

def plotting_3D(Amount_of_plots,Vec,Names):
    
def plotting_3D(direction, name, color, figure_num):
   plt.figure(figure_num)
   if (direction.shape[0])>2:
       for i in range(direction.shape[0]):
           #label = '%s Input Dir %i' % (name, i+1)

           plt.subplot((direction.shape[0]), 1, i + 1)
           plt.title(name)
           plt.semilogx(w, direction[i, :], color)

   else:

       plt.subplot(211)
       plt.title(name)
       plt.semilogx(w, direction[0, :], color)

       plt.subplot(212)
       plt.title(name)
       plt.semilogx(w, direction[1, :], color)   
