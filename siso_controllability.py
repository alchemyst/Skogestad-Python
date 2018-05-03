"""
Created on 22 Mar 2013

@author: St Elmo Wilken
"""
from __future__ import print_function
import control as cn
import numpy as np
import matplotlib.pyplot as plt

"""
This module checks if your plant conforms to the SISO controllability rules
as explained in Skogestad chapter 5.14.
"""


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


def rule_one_two(S, Gd, R=1.1,  freq=np.arange(0.001, 1, 0.001)):
    """
    Parameters: S => sensitivity transfer function
                Gd => transfer function of the disturbance
                w => frequency range (mostly there to make the plots look nice
    You want the red dashed line above the blue solid line for adequate
    disturbance rejection.
    """
    mag_s, phase_s, freq_s = cn.bode_plot(S, omega=freq)
    plt.clf()
    inv_gd = 1/Gd
    mag_i, phase_i, freq_i = cn.bode_plot(inv_gd, omega=freq)
    plt.clf()
    unity = [1] * len(freq)
    inv_R = [1.0/R] * len(freq)

    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("Magnitude")

"""
Rule 3 and 4: |u| < 1 to ensure your input is in the allowed range
              given that |e| < 1

Rule 3: Input constraints arising from disturbances.
|G| > |Gd| - 1 for acceptable control where |Gd| > 1

Rule 4: Input constraints arising from set points.
|G| > R - 1 for acceptable control
"""


def rule_three_four(G, Gd, R=1.1, perfect=True,
                    freq=np.arange(0.001, 1, 0.001)):
    """
    Parameters: G => transfer function of system
                Gd => transfer function of disturbances on system
                R => max allowed reference change relative to e (R > 1)
                perfect => Boolean: True then assumes perfect control
    """
    mag_g, phase_g, freq_g = cn.bode_plot(G, omega=freq)
    plt.clf()
    mag_gd, phase_gd, freq_gd = cn.bode_plot(Gd, omega=freq)
    plt.clf()

    plt.subplot(211)

    if perfect:
        plt.loglog(freq_gd, mag_g, color="red", label="|G|", ls="--")
        plt.loglog(freq_gd, mag_gd, color="blue", label="|Gd|")
    else:
        gd_min = mag_gd - 1
        gd_min = [x for x in gd_min if x > 0]
        freq_all = [
            freq_gd[x] for x in range(len(freq_gd)) if (mag_gd[x]-1) > 0]
        mag_g_all = [
            mag_g[x] for x in range(len(freq_gd)) if (mag_gd[x]-1) > 0]
        plt.loglog(freq_all, gd_min, color="blue", label="|Gd| - 1")
        plt.loglog(freq_all, mag_g_all, color="red", label="|G|", ls="--")

    plt.legend(loc=3)
    plt.grid()
    plt.ylabel("Magnitude")
    plt.xlabel("Frequency [rad/s]")

    plt.subplot(212)
    plt.loglog(freq_g, mag_g, color="red", label="|G|", ls="--")
    if perfect:
        R_stretch = len(freq_g)*[np.abs(R)]
        plt.loglog(freq_g, R_stretch, color="blue", label="|R|")
    else:
        R_stretch = np.subtract(len(freq_g)*[np.abs(R)], [1]*len(freq_g))
        plt.loglog(freq_g, R_stretch, color="blue", label="|R|-1")
    plt.legend(loc=1)
    plt.grid()
    plt.ylabel("Magnitude")
    plt.xlabel("Frequency [rad/s]")


"""
Rules 5 - 8:
These are more like bounds on the bandwidth given certain characteristics of
the system.
They do depend on the bound of either S or T...
(S for zeros, T for poles via internal stability)
The important thing is really just Theorem 5.3 + the interpolation
constraints... everything else follows from that.
"""


"""
This method will just show the intersection of the dead time frequency
with the loop gain magnitude (forms part of rule 1 and rule 5).

Effectively what you want here is to know if:
|Gd(w_theta)| < 1 where w_theta = 1/(dead time)
"""


def dead_time_bound(L, Gd, deadtime, freq=np.arange(0.001, 1, 0.001)):
    """
    Parameters: L => the loop transfer function
                Gd => Disturbance transfer function
                deadtime => the deadtime in seconds of the system
    Notes: If the cross over frequencies are very large or very small
    you will have to find them yourself.
    """
    mag, phase, omega = cn.bode_plot(L, omega=freq)
    mag_d, phase_d, omega_d = cn.bode_plot(Gd, omega=freq)
    plt.clf()

    gm, pm, wg, wp_L = cn.margin(mag, phase, omega)
    gm, pm, wg, wp_Gd = cn.margin(mag_d, phase_d, omega_d)

    freq_lim = [freq[x] for x in range(len(freq)) if 0.1 < mag[x] < 10]
    mag_lim = [mag[x] for x in range(len(freq)) if 0.1 < mag[x] < 10]

    plt.loglog(freq_lim, mag_lim, color="blue", label="|L|")

    dead_w = 1.0/deadtime
    ymin, ymax = plt.ylim()
    plt.loglog([dead_w, dead_w], [ymin, ymax], color="red", ls="--",
               label="dead time frequency")
    plt.loglog([wp_L, wp_L], [ymin, ymax], color="green", ls=":",
               label="w_c")
    plt.loglog([wp_Gd, wp_Gd], [ymin, ymax], color="black", ls="--",
               label=" w_d")
    print("You require feedback for disturbance rejection up to (w_d) = " +
          str(wp_Gd) +
          "\n Remember than w_B < w_c < W_BT and  w_d < w_B hence w_d < w_c.")
    print("The upper bound on w_c based on the dead time\
           (wc < w_dead = 1/dead_seconds) = " + str(1.0/deadtime))

    plt.legend(loc=3)
