'''
Created on 21 Mar 2013

@author: St Elmo Wilken
'''

import control as cn
import numpy as np
import matplotlib.pyplot as plt

def plot_sensitivity(sys, *args, **dict):
    """
    Plots the bode graphs of the sensitivity function = 1/(1 + L)
    Currently works for SISO only.
    """  
    foo = cn.tf([1],[1])
    bar = cn.parallel(foo, sys) # equivalent to 1 + L
    num, den = cn.tfdata(bar)
    foo = cn.tf(den, num) # equivalent to 1/(1+L)
    mag, phase, omega = cn.bode_plot(foo, *args, **dict)
    return mag, phase, omega
    
def plot_complement_sensitivity(sys, *args, **dict):
    """
    Plots the bode graphs of the complementary sensitivity function = L/(1+L)
    Currently works for SISO only.
    """
    foo = cn.tf([1],[1])
    bar = cn.parallel(foo, sys) # equivalent to 1 + L
    num, den = cn.tfdata(bar)
    foo = cn.tf(den, num) # equivalent to 1/(1+L)
    bar = cn.series(foo, sys) #equivalent to L* 1/(1+L)
    mag, phase, omega = cn.bode_plot(bar, *args, **dict)
    return mag, phase, omega
    
def peaks(sys):
    """
    Returns the H infinity norms and the corresponding frequencies for
    the sensitivity and complementary sensitivity functions.
    Currently works for SISO only.
    """
    plt.figure(99)
    mag_S, phase_S, omega_S = plot_sensitivity(sys)
    mag_T, phase_T, omega_T = plot_complement_sensitivity(sys)
    plt.close(99) #so that this does not interfere with your current plotting ;)
    pos_S = np.argmax(mag_S)
    pos_T = np.argmin(mag_T)
    s_data = ("Peak |S|", mag_S[pos_S], phase_S[pos_S])
    t_data = ("Peak |T|", mag_T[pos_T], phase_T[pos_T])
    return s_data, t_data

   
    