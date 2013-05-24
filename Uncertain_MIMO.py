import numpy as np
import matplotlib.pyplot as plt
import matplotlib as ml
import scipy.optimize as op

ml.rcParams['font.family'] = 'serif'
ml.rcParams['font.serif'] = ['Cambria'] + ml.rcParams['font.serif']

# Calculates plant perturbations
# Currently for SISO only

# Give the minimum and maximum values of a spesific variable in a matrix form
# For example a matrix of the minimum and maximum values of
# a certain set of parameters

# Create a matrix of all possibilities of the umin and umax vectors
# The first entry of the matrix correspondse to the first entry
# in the minimum and maximum matrices


def bound_SISO_wi(w_start, w_end, vec, G, Gp, steps):

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
    return Gp_i_max


def weight_calc(w_start, w_end, li, weight_i, steps):
    """Calculates a simple third-order weight"""

    w = np.logspace(w_start, w_end, steps)

    # Search for the first index where the next value is lower
    # than the previous
    # Only works if no peaks are present and a low order
    # weight is to be fitted

    init_old = 0
    for k in range(steps):
        init = li[k]
        if init < init_old:
            index = k
            break
        else:
            init_old = init
    max_val = np.max(li[index:-1])

    # Write the maximum value over the rest of the range
    l_mod = li
    z = 0  # margin on turn
    for k in range(steps - index - z):
        i = k + z + index
        l_mod[i] = max_val

    # Must change a, b, and c in weight to fit to l_mod

    popt, pcov = op.curve_fit(weight_i, w, l_mod)

    return popt, l_mod, w

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