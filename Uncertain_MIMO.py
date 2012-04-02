import numpy as np
import scipy.linalg as sc_lin
import matplotlib.pyplot as plt


# start deriving and implementing equation for uncertainty


def uncertian_MIMO():

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
