import numpy as np
import scipy.linalg as sc_lin
import matplotlib.pyplot as plt
import itertools
from perturbations import possibilities

# Calculates plant perturbations
# Currently for SISO only

# Give the minimum and maximum values of a spesific variable in a matrix form
# For example a matrix of the minimum and maximum values of
# a certain set of parameters

# Create a matrix of all possibilities of the umin and umax vectors
# The first entry of the matrix correspondse to the first entry
# in the minimum and maximum matrices


def bound_SISO_wi(w_start, w_end, vec, weight, G, Gp, steps):

    w = np.logspace(w_start, w_end, steps)

#    Gp_i = np.zeros((vec.shape[0], len(w), 1))
    Gp_i_max = np.zeros((len(w)))

    # TODO: Calculate w_i for spesific plant with a set amount of perturbations
    #       for multiplicative uncertainty

    # Plotting the multiplicative relative uncertianty as a function of w
    
    # Calculate sequences to plot all perturbations
#    for i in range(vec.shape[0]):
#        Gp_i = [(Gp(w_i * 1j, vec, i) - G(w_i * 1j)) / (G(w_i * 1j))
#                for w_i in w]
#        plt.loglog(w, np.abs(Gp_i), 'b+')
#
    # Calculate the vales by frequency and store the highest 
    # value at each frequency
    # Plotting only the maximum value evaluated at each frequency
        
    for k in range(len(w)):
        w_i = w[k]
        Gp_i = [(Gp(w_i * 1j, vec, i) - G(w_i*1j)) / (G(w_i * 1j))
                for i in range(vec.shape[0])]
        Gp_i_max[k] = np.max(np.abs(Gp_i))
#    plt.loglog(w, Gp_i_max, 'b+')

    # Choice of wieght function for uncertainty
#    weight_function = [weight(w_i * 1j) for w_i in w]
#    plt.loglog(w, np.abs(weight_function), 'g')

#    plt.show()
    return Gp_i_max


    # TODO: The following is a list of possible expansions
    # Calculate upper bounds on S'
    # eq 6-88 pg 248

    # Check input and output condition number along with RGA elements

    # Nuquist plot with perturbed plant models generating all
    # possible plants Nuquist plots using method outlined in 7.4.3
    # pg 268-271 Skogestad


    #################
    # Use of state space descriptions
    # Easier to discribe uncertainty and is numerically easier
    #################

    ####
    # Higher frequency uncertainty (higher order weights)
    # For example see eq 7-36 and 7-37
    ####