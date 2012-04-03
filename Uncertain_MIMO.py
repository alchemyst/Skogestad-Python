import numpy as np
import scipy.linalg as sc_lin
import matplotlib.pyplot as plt


# start deriving and implementing equation for uncertainty this is for SISO systems 


def G(s):
    """returns value of the nominal plant"""
    G= 
    return G 


def Gp(s,vec,i):
    """vec contains all perturbations of the varaibles to be used
    i is the counter to be used
    Gp is the perturbed plant"""

    Gp= 
    return Gp

def Bound_SISO_wi():

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
