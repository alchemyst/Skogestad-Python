from __future__ import division
from __future__ import print_function
from builtins import range

import numpy as np
import matplotlib as ml
import scipy.optimize as op

ml.rcParams['font.family'] = 'serif'
ml.rcParams['font.serif'] = ['Cambria'] + ml.rcParams['font.serif']

# Calculates plant perturbations
# Currently for SISO only

# Give the minimum and maximum values of a specific variable in a matrix form
# For example a matrix of the minimum and maximum values of
# a certain set of parameters

# Create a matrix of all possibilities of the umin and umax vectors
# The first entry of the matrix corresponds to the first entry
# in the minimum and maximum matrices


def bound_SISO_wi(w_start, w_end, vec, G, Gp, steps):

    w = np.logspace(w_start, w_end, steps)

#    Gp_i = np.zeros((vec.shape[0], len(w), 1))
    Gp_i_max = np.zeros((len(w)))

    # Plotting the multiplicative relative uncertianty as a function of w
    # Plotting the multiplicative relative uncertainty as a function of w


    # Calculate sequences to plot all perturbations
#    for i in range(vec.shape[0]):
#        Gp_i = [(Gp(w_i * 1j, vec, i) - G(w_i * 1j)) / (G(w_i * 1j))
#                for w_i in w]
#        plt.loglog(w, np.abs(Gp_i), 'b+')
#        plt.ylabel('Deviation due to uncertainty')
#        plt.xlabel('Frequency (rad/min)')
#
    # Calculate the vales by frequency and store the highest
    # value at each frequency
    # Plotting only the maximum value evaluated at each frequency

    for k in range(len(w)):
        w_i = w[k]
        Gp_i = [(Gp(w_i * 1j, vec, i) - G(w_i * 1j)) / (G(w_i * 1j))
                for i in range(vec.shape[0])]
        Gp_i_max[k] = np.max(np.abs(Gp_i))

#    plt.loglog(w, Gp_i_max, 'b+')
    return Gp_i_max


def bound_MIMO_wi(w_start, w_end, vec, G, Gp, steps):

    w = np.logspace(w_start, w_end, steps)

#    Gp_i = np.zeros((vec.shape[0], len(w), 1))
    Gp_i_max = np.zeros((len(w)))

    # Plotting the multiplicative output relative uncertianty
    # Plotting the multiplicative output relative uncertainty
    # as a function of w

    # Calculate the vales by frequency and store the highest
    # value at each frequency
    # Plotting only the maximum value evaluated at each frequency

    for k in range(len(w)):
        w_i = w[k]
        # Calculate all perturbations at a specific frequency
        Gp_i = [np.linalg.svd((Gp(w_i * 1j, vec, i) - G(w_i * 1j))
                * np.linalg.pinv(G(w_i * 1j)), compute_uv=False)[0]
                for i in range(vec.shape[0])]
        # Calculate the maximum of the maximum singular values
        # at each frequency
        Gp_i_max[k] = np.max(Gp_i)
#    plt.loglog(w, Gp_i_max, 'b+')
    return Gp_i_max


def weight_calc(w_start, w_end, li, weight_i, steps):
    """Calculates a simple third-order weight
    Accommodates situations were the weight increases with frequency
    """

    w = np.logspace(w_start, w_end, steps)

    # Search for the first index where the next value is lower
    # than the previous
    # Only works if no peaks are present and a low order
    # weight is to be fitted

    init_old = 0
    found = False
    safety_fac = 10  # amount of initial indexes to ignore
    for k in range(steps - safety_fac):
        init = li[k + safety_fac]
        if init < init_old:
            index = k + safety_fac
            found = True
            print(index)
            break
        else:
            init_old = init
    l_org = np.copy(li)
    l_mod = li
    if found:
        max_val = np.max(li[index:-1])
        # Write the maximum value over the rest of the range
        z = 0  # margin on turn
        for k in range(steps - index - z):
            i = k + z + index
            l_mod[i] = max_val

    # Must change a, b, and c in weight to fit to l_mod

    popt, pcov = op.curve_fit(weight_i, w * 1j, l_mod, [0.5, 0.5, 0.5])

    return popt, l_mod, l_org, w


def weight_calc_dec(w_start, w_end, li, weight_i, steps):
    """Calculates a simple third-order weight
    Accommodates situations were the weight decreases with frequency
    """

    w = np.logspace(w_start, w_end, steps)

    # Search for the first index where the next value is lower
    # than the previous
    # Only works if no peaks are present and a low order
    # weight is to be fitted

#    init_old = 0
#    found = False
#    safety_fac = 10  # amount of initial indexes to ignore
#    for k in range(steps - safety_fac):
#        init = li[k + safety_fac]
#        if init > init_old:
#            index = k + safety_fac
#            found = True
#            print index
#            break
#        else:
#            init_old = init
    l_org = np.copy(li)
    l_mod = li
#    if found:
#        max_val = np.max(li[index:-1])
#        # Write the maximum value over the rest of the range
#        z = 0  # margin on turn
#        for k in range(steps - index - z):
#            i = k + z + index
#            l_mod[i] = max_val


    # Must change a, b, and c in weight to fit to l_mod

    popt, pcov = op.curve_fit(weight_i, w * 1j, l_mod, [0.5, 0.5, 0.5])

    return popt, l_mod, l_org, w

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
    # Easier to describe uncertainty and is numerically easier
    #################

    ####
    # Higher frequency uncertainty (higher order weights)
    # For example see eq 7-36 and 7-37
    ####
