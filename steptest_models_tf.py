# -*- coding: utf-8 -*-
"""
@author: Simon Streicher
"""

import numpy as np
from utils import tf

"""
This file contains all the transfer function models that are used in the
report titled "Controllability analysis of a depropaniser column"

Models added to repository to demonstrate working of steptests function.
"""

# Gain of first stabilising feedback controller
Kc1 = 1 / -2.0  # 2.0
# Gain of second stabilising feedback controller
Kc2 = 1 / -4.0  # 0.088


# Define functions using tf object in utils

s = tf([1, 0])


def G_unscaled():
    """
    Gives the process transfer function matrix
    Deadtime is defined separately

    Outputs according to rows (first index)
    Inputs according to columns (second index)

    Inputs:
        1: Reboiler condensate flow
        2: Reflux flow rate
        3: Overhead gas product flow rate
        4: Bottoms flow rate

    Outputs:
        1: Bottoms temperature
        2: Column top pressure
        3: Reboiler level
        4: Overhead receiver level
    """
    # Bottom temperature vs reboiler condensate flow
    g11 = 2.75e1 * (20.0 * s + 1) / ((10.0 * s + 1) ** 4)
    # Bottom temperature vs reflux flow rate
    g12 = -2.73e-1 / (14.0 * s + 1) ** 2
    # Bottom temperature vs overhead gas product flow rate
    g13 = 0.0 * s
    # Bottom temperature vs bottoms flow rate
    g14 = 0.0 * s

    g21 = 1.66 / (16.6 * s + 1)
    g22 = 1.47e-1 / (3.64e1 * s + 1)
    g23 = -3.02e-3 / ((2.27e1 * s + 1) * (6.17 * s + 1))
    g24 = 0.0 * s

    g31 = (-10.97 / (-2.0 * Kc1)) / ((1 / (-2.0 * Kc1)) * s + 1)
    g32 = (1.275 / (-2.0 * Kc1)) / ((1 / (-2.0 * Kc1)) * s + 1)
    g33 = 0.0 * s
    g34 = (-2.0 / (-2.0 * Kc1)) / ((1 / (-2.0 * Kc1)) * s + 1)

    g41 = (0.592 / (-0.088 * Kc2)) / ((1 / (-0.088 * Kc2)) * s + 1)
    g42 = (-0.088 / (-0.088 * Kc2)) / ((1 / (-0.088 * Kc2)) * s + 1)
    g43 = 0.0 * s
    g44 = 0.0 * s

    G = np.matrix([[g11, g12, g13, g14],
                   [g21, g22, g23, g24],
                   [g31, g32, g33, g34],
                   [g41, g42, g43, g44]])
    return G


def Gd_unscaled():
    """
    Gives the disturbance transfer function
    Deadtime is defined separately

    Outputs according to rows (first index)
    Inputs according to columns (second index)

    Only a single input is present:
        Total feed rate

    Outputs are the same as for the process transfer function matrix
    """

    gd11 = -2.24 / ((109 * s + 1.0) * (3.12 * s + 1.0))
    gd21 = 0.0
    gd31 = (2.29 / (-2.0 * Kc1)) / ((1 / (-2.0 * Kc1) * s + 1))
    gd41 = 0.0

    Gd = np.matrix([[gd11],
                    [gd21],
                    [gd31],
                    [gd41]])
    return Gd


def Zeros_Poles_RHP():
    """
    Give a vector with all the RHP zeros and poles
    RHP zeros and poles are calculated from Sage / MatLab program
    """

    Zeros_G = [5.6612]
    Poles_G = [0]

    return Zeros_G, Poles_G


def dead_time():
    """
    Vector of the deadtime of the system
    """
    # Individual time delays
    dead_G = np.matrix([[15.0, 15.0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [15.0, 15.0, 0, 0]])
    dead_Gd = np.matrix([[0],
                         [0],
                         [4.45],
                         [0]])

    return dead_G, dead_Gd


def De():
    """
    Gives the error scaling matrix De
    """
    de11 = 2.0
    de22 = 0.5
    de33 = 20
    de44 = 20

    De = np.matrix([[de11, 0, 0, 0],
                    [0, de22, 0, 0],
                    [0, 0, de33, 0],
                    [0, 0, 0, de44]])
    return De


def Du():
    """
    Gives the input scaling matrix Du
    """
    du11 = 1.5
    du22 = 12
    du33 = 1500
    du44 = 15

    Du = np.matrix([[du11, 0, 0, 0],
                    [0, du22, 0, 0],
                    [0, 0, du33, 0],
                    [0, 0, 0, du44]])
    return Du


def Dd():
    """
    Gives the disturbance scaling matrix Dd
    """
    dd11 = 10

    Dd = np.matrix([[dd11]])
    return Dd


def G_unscaled_dt():
    """
    Creates a process transfer matrix with deadtime included
    """

    G_deadtime = np.zeros((np.shape(G_unscaled())[0],
                           np.shape(G_unscaled())[1]), dtype=object)
    for i in range(np.shape(G_unscaled())[0]):
        for k in range(np.shape(G_unscaled())[1]):
            G_deadtime[i, k] = G_unscaled()[i, k]
            G_deadtime[i, k].deadtime = dead_time()[0][i, k]

    return G_deadtime


def Gd_unscaled_dt():
    """
    Creates a disturbance transfer matrix with deadtime included
    """

    Gd_deadtime = np.zeros((np.shape(Gd_unscaled())[0],
                            np.shape(Gd_unscaled())[1]),
                           dtype=object)
    for i in range(np.shape(Gd_unscaled())[0]):
        for k in range(np.shape(Gd_unscaled())[1]):
            Gd_deadtime[i, k] = G_unscaled()[i, k]
            Gd_deadtime[i, k].deadtime = dead_time()[1][i, k]

    return Gd_deadtime


def G_nodeadtime():
    """
    Gives the scaled process transfer matrix without deadtime
    """

    G_s = De().I * G_unscaled() * Du()
    return G_s


def G():
    """
    Gives the scaled process transfer matrix with deadtime
    """

    G = np.zeros((np.shape(G_nodeadtime())[0],
                  np.shape(G_nodeadtime())[1]), dtype=object)
    for i in range(np.shape(G_nodeadtime())[0]):
        for k in range(np.shape(G_nodeadtime())[1]):
            G[i, k] = G_nodeadtime()[i, k]
            G[i, k].deadtime = dead_time()[0][i, k]

    return G


def Gd_nodeadtime():
    """
    Gives the scaled disturbance transfer matrix without deadtime
    """

    Gd_s = De().I * Gd_unscaled() * Dd()
    return Gd_s


def Gd():
    """
    Gives the scaled disturbance transfer matrix with deadtime
    """

    Gd_s = De().I * Gd_unscaled_dt() * Dd()
    return Gd_s