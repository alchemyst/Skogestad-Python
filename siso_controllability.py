'''
Created on 22 Mar 2013

@author: St Elmo Wilken
'''

"""
This module checks if your plant conforms to the SISO controllability rules
as explained in Skogestad chapter 5.14.
"""

import control as cn
import numpy as np
import matplotlib.pyplot as plt
import control_add_on as cadd

"""
Rule 1 and 2: |e| < 1 for acceptable control given feedback

Rule 1: Speed of response to reject disturbances.
How fast must the plant be to be able to reject disturbances?
e = S*Gd*d
If you require |e| < 1 and given the worst case disturbance i.e. |d| = 1
then |S*Gd| < 1 (assuming feedback control is used). 

Rule 2: Speed of response to track reference changes.
Similar to rule 1 but this time you want |S| < 1\R where
R is the max allowed reference change relative to the allowed error.
"""
def rule_one(S, Gd, R,  freq = np.arange(0.001, 1,0.001)):
    """
    Parameters: S => sensitivity transfer function
                Gd => transfer function of the disturbance
                w => frequency range (mostly there to make the plots look nice
    You want the red dashed line above the blue solid line for adequate disturbance rejection.
    """
    mag_s, phase_s, freq_s = cn.bode_plot(S, omega = freq)
    plt.clf()
    inv_gd = 1/Gd
    mag_i, phase_i, freq_i = cn.bode_plot(inv_gd, omega = freq)
    plt.clf()
    unity = [1]*len(freq)
    inv_R = [1/R]*len(freq)
    
    plt.loglog(freq_s, inv_R, color = "green", lw = 3.0, ls = "--", label = "1/R")
    plt.loglog(freq_s, unity, color ="black", ls = ':')
    plt.loglog(freq_s, mag_s,  color = "blue", label = "|S|")
    plt.loglog(freq_i, mag_i, color = "red", ls = '--', label = "1/|Gd|")
    plt.legend(loc = 4)
    plt.grid()
    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("Magnitude")

"""
Rule 3 and 4: |u| < 1 to ensure your input is in the allowed range

Rule 3: Input constraints arising from disturbances.

Rule 4: Input constraints arising from setpoints.
"""



"""Rule 5:  Time delay."""

"""Rule 6: Tight control at low frequencies with an RHP zero."""

"""Rule 7: Not part of the syllabus..."""

"""Rule 8: Real open loop unstable pole."""


