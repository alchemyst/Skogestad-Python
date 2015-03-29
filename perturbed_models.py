# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:13:53 2013

@author: Simon Streicher
"""

import matplotlib.pyplot as plt
import matplotlib as ml
import numpy as np

from uncertain_MIMO import bound_SISO_wi, weight_calc
from perturbations import possibilities


ml.rcParams['font.family'] = 'serif'
ml.rcParams['font.serif'] = ['Cambria'] + ml.rcParams['font.serif']

"""Calculates pertubated models for specifc elements"""

"""This section is for the g11 element in the process transfer matrix"""

# Define bounds on each uncertain parameter

"""[k, theta, alpha, beta, gamma]"""

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
    """Returns value of the nominal plant"""
    G = 20.6 * (20 * s + 1) * np.exp(-1 * s * 15) / ((10.0 * s + 1) ** 4)
    return G


def Gnp(s):
    """Returns factor part of nominal plant that is not perturbed"""
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


def weight_i(s, a, b, c):
    """Function to give the weight of the uncertainty"""

    w_i = (((s * a + start) / ((a /end) * s + 1))
           * ((s ** 2 + b * s + 1) / (s ** 2 + c * s + 1)))
#    w_i = ((((s * a + start) / ((a / (end)) * s + 1))
#            * ((s ** 2 + b * s + 1) / (s ** 2 + c * s + 1)))
#           * ((s ** 2 + d * s + 1) / (s ** 2 + e * s + 1)))
    return w_i

# Call bound function
li = bound_SISO_wi(w_start, w_end, vec, G, Gp, steps)

start = np.min(li)
end = np.max(li)

# Call weight calculation function
popt, l_mod, w = weight_calc(w_start, w_end, li, weight_i, steps)

# Modify the weight parameters to get a better fit
popt_mod = popt
popt_mod[0] = popt[0] + 0.5
popt_mod[1] = popt[1]
popt_mod[2] = popt[2]

# Plot the weight and the li error it must be above
# The red line must be above the blue line at all time
plt.loglog(w, l_mod, 'b')
plt.loglog(w, weight_i(w, popt_mod[0], popt_mod[1], popt_mod[2]), 'r')
plt.show()
