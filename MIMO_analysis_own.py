import numpy as np
import scipy as sc
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from mpl_toolkits.mplot3d import Axes3D
from Perturbations import Possibilities 

#drawing circles in all directions with G and Gd to see were problem could be expected and in what direction

#geving perturbations matrices for use in creud uncertainty analysis of a MIMO system with disturbance matrix 

umin_G=np.matrix([[0.8],[0.08],[0.8],[2.6],[8],[0.8],[80],[0.8],[1.6],[2.4],[24]])
umax_G=np.matrix([[1.2],[0.12],[1.2],[3.4],[12],[1.2],[120],[1.2],[2.4],[3.6],[36]])
print umin_G.shape

umin_Gd=np.matrix([[0.8],[0.8],[1.6],[1.6],[3.4],[0.8],[6],[3.4],[1.6],[0.8],[8],[3.4],[0.8],[4.2],[2.8]])
umax_Gd=np.matrix([[1.2],[1.2],[2.4],[2.4],[4.4],[1.2],[10],[4.6],[2.4],[1.2],[10],[4.6],[1.2],[5.6],[3.2]])
print umin_Gd.shape

def Gd(s):
    """disturbance transfer function matrix 
    needs to have the same amount of columns than G"""
    Gd= np.matrix([[1/(s+1), s**2/((s+2)*(s+4)), (s+1)/((s+8)*(s+4)*(s+2))], [1/(s+10), 4, s*(s+1)/((s+5)*(s+3))]])
    return Gd

def Gdp(s, vec):
    """disturbance transfer function matrix as a function of perturbations
    needs to have the same size as Gd matrix"""

    #needs vec of perturbations that is 15 long 
    Gd= np.matrix([[vec[:, 0]/(s+vec[:, 1]), s**vec[:, 2]/((s+vec[:, 3])*(s+vec[:, 4])), (s+vec[:, 5])/((s+vec[:, 6])*(s+vec[:, 7])*(s+vec[:, 8]))], [vec[:, 9]/(s+vec[:, 10]), vec[:, 11], s*(s+vec[:, 12])/((s+vec[:, 13])*(s+vec[:, 14]))]])
    return Gdp


def G(s):
    """system transfer function matrix
    needs to have the same amount of columns than Gd"""
    G =np.matrix([[1/(s+0.1), (s+1)/((s+3)*(s+10))], [1/(s+100), (s+1)*(s+2)/((s+3)*(s+30))]])
    return G

def Gp(s, vec):
    """system transfer function matrix as a function of perturbations 
    needs to be the same size as G"""

    #needs perturbation vector of length of 11
    G =np.matrix([[vec[:, 0]/(s+vec[:, 1]), (s+vec[:, 2])/((s+vec[:, 3])*(s+vec[:, 4]))], [vec[:, 5]/(s+vec[:, 6]), (s+vec[:, 7])*(s+vec[:, 8])/((s+vec[:, 9])*(s+vec[:, 10]))]])
    return Gp

def directions(Amount, w_start, w_end, umin_G, umin_Gd, umax_G, umax_Gd, Amount_Plant):
    """give all possible directions of a certain transfer function matrix
    to be used with SVD to plot the magnitude and direction in all directions
    This would give indication where control problems would occurs"""

    #next section assumes correct scaling of G and Gd
    #generating all possible input directions for Gd
    umin_Gd_input = np.matrix(np.zeros([Gd(0.001).shape[1], 1]))
    umax_Gd_input = np.matrix(np.ones([Gd(0.001).shape[1], 1]))
    Possibilities_plant_Gd = Possibilities(umin_Gd_input, umax_Gd_input, Amount)

    #generating all possible input direction for G
    umin_G_input = np.matrix(np.zeros([G(0.001).shape[1], 1]))
    umax_G_input = np.matrix(np.ones([G(0.001).shape[1], 1]))
    Possibilities_plant_G = Possibilities(umin_G_input, umax_G_input, Amount)
    #print Possibilities_plant_G
    
    #generating all possible matrix containing all perturbations for Gd matrix and G matrix
    perturbation_Gd_parameters = Possibilities(umin_Gd, umax_Gd, Amount_Plant)
    perturbation_G_parameters = Possibilities(umin_G, umax_G, Amount_Plant)

    print perturbation_G_parameters.shape
    
    w_iter=np.logspace(w_start, w_end, 100)

    #creating matrices for storing of all the data
    G_output = np.zeros([len(w_iter)*np.shape(perturbation_G_parameters)[0]*Possibilities_plant_G.shape[0],np.shape(G(0.0001))[0]])
    Gd_output = np.zeros([len(w_iter)*np.shape(perturbation_Gd_parameters)[0]*Possibilities_plant_Gd.shape[0],np.shape(Gd(0.0001))[0]])

    
    
    #calculating the directions and magnitude from G and Gd from all possible input directions and freqeuncies 
    #G calculations first 
    #for all freqeuncies 
    #for all perturbations
    #calculate each direction  of G and Gd output to y 
    #store directions of G and Gd 
    #restore freqeuncy vector 
    #store magnitude of G and Gd of y 
    #for w_i in range(len(w_iter)):
    #    for j_pos in range(Possibilities_plant_G.shape[0]):


#directions(Amount, w_start, w_end, Gd_perturbations_ready, G_perturbation_ready, umin_G, umin_Gd, umax_G, umax_Gd, Amount_Plant):
directions(10, -3, 3, umin_G, umin_Gd, umax_G, umax_Gd, 2)




