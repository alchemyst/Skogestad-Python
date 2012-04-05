import numpy as np
import scipy.linalg as sc_lin
import matplotlib.pyplot as plt
import itertools
from Perturbations import Possibilities

# start deriving and implementing equation for uncertainty this is for SISO systems

#give the minimum and maximum values of a spesific variable in a matrix form

#for example a matrix of the minimum and maximum values of a certain set of parameters
umin     =[[1], [1], [1]]
umax     =[[3], [3], [3]]

def G(s):
    """returns value of the nominal plant"""
    G=2.5/(2.5*s+1)
    return G


def Gp(s, vec, i):
    """vec contains all perturbations of the varaibles to be used
    i is the counter to be used
    Gp is the perturbed plant"""

    Gp = (vec[i, 0]/(vec[i, 1]*s+1))*np.exp(-1*s*vec[i, 2])
    return Gp

def weight_i(s):
    """function to give the weight of the uncertainty"""
    w_i = (4*s+0.5)/((4/2.5)*s+1)
    return w_i


#create a matrix of all possibilities of the umin and umax vectors
#the first entry of the matrix correspondse to the first entry in the minimum and maximum matrices
vec = Possibilities(umin, umax, 5)


def Bound_SISO_wi(w_star, w_end, vec):
    #upper bounds on S'
    #eq6-88 pg 248

    #check input and output condition number along with RGA elements


    #nuquist plot with perturbed plant models generating all possible plants nuquist plots
    #pg 268-271 skogestad
    #using method outlined in 7.4.3

    #use mesh function from CSC research project to subpliment usample function of matlab

    #################
    #most probably going to use State Space , easier to discribe uncertainty and is numerically easier
    #################

    ####
    #eq 7-36 and 7-37
    #higher frequency uncertainty
    ####

    ####
    #calculating w_i for spesific plant with a set amount of perturbations
    #multiplicative uncertainty
    ####
    w=np.logspace(w_star, w_end, 100)

    #plotting the multiplicative relative uncertianty as a function of w
    for i in range(vec.shape[0]):
        Gp_multiplicative = [(Gp(w_i*1j, vec, i)-G(w_i*1j))/(G(w_i*1j)) for w_i in w]
        plt.loglog(w, np.abs(Gp_multiplicative), 'b+')

    #choice of wieght function for uncertainty
    weight_function =[weight_i(w_i*1j) for w_i in w]
    plt.loglog(w, weight_function)

    plt.show()


Bound_SISO_wi(-2, 2, vec)
