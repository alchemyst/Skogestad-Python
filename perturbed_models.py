# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:13:53 2013

@author: Simon Streicher
"""

import matplotlib.pyplot as plt
import matplotlib as ml
import numpy as np
from uncertain_MIMO import bound_SISO_wi
from perturbations import possibilities
import scipy.optimize as op

ml.rcParams['font.family'] = 'serif'
ml.rcParams['font.serif'] = ['Cambria'] + ml.rcParams['font.serif']

"""
Calculates pertubated models for specifc elements
"""

"""
This section is for the g11 element in the process transfer matrix
"""

# Define bounds on each uncertain parameter

"""
[k, theta, alpha, beta, gamma]
"""

umin = [0.8, 13, 0.95, 0.95, 0.95]
umax = [1.2, 17, 1.05, 1.05, 1.05]
#umin = [1, 15, 1, 1, 1]
#umax = [1, 15, 1, 1, 1]

# Define amount of steps to use
steps = 100


# Define start and end of frequency investigated
w_start = -3
w_end = 2

vec = possibilities(umin, umax, 3)


def G(s):
    """
    Returns value of the nominal plant
    """
#    G = 20.6 * (20 * s + 1) / ((10.0 * s + 1) ** 4)
    G = 20.6 * (20 * s + 1) * np.exp(-1 * s * 15) / ((10.0 * s + 1) ** 4)
    return G


def Gnp(s):
    """
    Returns factor part of nominal plant that is not perturbed
    """
    Gnp = 20.6 / (10.0 * s + 1) ** 2
    return Gnp


def Gp(s, vec, i):
    """
    Vec contains all perturbations of the varaibles to be used
    i is the counter to be used
    Gp is the perturbed plant
    """
#    Gp = Gnp(s) * ((vec[i, 0] * (vec[i, 4] * 20 * s + 1))
#                    / ((vec[i, 2] * 10 * s + 1)
#                   * (vec[i, 3] * 10 * s + 1)))
    Gp = Gnp(s) * ((vec[i, 0] * (vec[i, 4] * 20 * s + 1))
                   / ((vec[i, 2] * 10 * s + 1)
                   * (vec[i, 3] * 10 * s + 1))
                   * np.exp(-1 * s * vec[i, 1]))
    return Gp

# Call bound function


def weight_i(s, a, b, c):
    """
    Function to give the weight of the uncertainty
    """
    w_i = (((s * a + start) / ((a / (end)) * s + 1))
           * ((s ** 2 + b * s + 1) / (s ** 2 + c * s + 1)))
#    w_i = ((((s * a + start) / ((a / (end)) * s + 1))
#            * ((s ** 2 + b * s + 1) / (s ** 2 + c * s + 1)))
#           * ((s ** 2 + d * s + 1) / (s ** 2 + e * s + 1)))
    return w_i

li = bound_SISO_wi(-3, 2, vec, weight_i, G, Gp, steps)[:]
start = np.min(li)
end = np.max(li)


w = np.logspace(w_start, w_end, steps)

# Define range over which maximum value must be searched

# Search for the first index where the next value is lower than the previous
# Only works if no peaks are present and a low order weight is to be fitted

init_old = 0
for k in range(steps):
    init = li[k]
    if init < init_old:
        index = k
        break
    else:
        init_old = init


max_val = np.max(li[index:-1])

# Write that value in that range

l_mod = li
z = 0  # margin on turn
for k in range(steps - index - z):
    i = k + z + index
    l_mod[i] = max_val

#plt.loglog(w, l_mod, 'b+')
#plt.show()

# Must change a, b, and c in weight to fit to l_mod


popt, pcov = op.curve_fit(weight_i, w, l_mod)

# Modify the weight parameters to get a better fit
popt_mod = popt
popt_mod[0] = popt[0] + 0.5
popt_mod[1] = popt[1]
popt_mod[2] = popt[2]

plt.loglog(w, l_mod, 'b')
plt.loglog(w, weight_i(w, popt_mod[0], popt_mod[1], popt_mod[2]), 'g')
plt.show()








