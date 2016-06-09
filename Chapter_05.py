from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import doc_func as df

from utils import tf, margins

"""
All of the below function are from pages 206-207 and the associated rules that
analyse the controllability of SISO problems
"""

# TO DO merge all siso_controllability.py routines with this file
def rule1(G, Gd, K=1, message=False, plot=False, w1=-4, w2=2):
    """
    This is rule one of chapter five

    Calculates the speed of response to reject distrurbances. Condition require
    |S(jw)| <= |1/Gd(jw)|

    Parameters
    ----------
    G : tf
        plant model

    Gd : tf
        plant distrubance model

    K : tf
        control model

    message : boolean
        show the rule message (optional)

    plot : boolean
        show the bode plot with constraints (optional)

    w1 : integer
        start frequency, 10^w1 (optional)

    w2 : integer
        end frequency, 10^w2 (optional)

    Returns
    -------
    valid1 : boolean
        value if rule conditions was met

    wc : real
        crossover frequency where | G(jwc) | = 1

    wd : real
        crossover frequency where | Gd(jwd) | = 1
    """

    _,_,wc,_ = margins(G)
    _,_,wd,_  = margins(Gd)

    valid1 = wc > wd

    if message:
        print('Rule 1: Speed of response to reject distrubances')
        if valid1:
            print('First condition met, wc > wd')
        else:
            print('First condition no met, wc < wd')
        print('Second condition requires |S(jw)| <= |1/Gd(jw)|')

    if plot:
        w = np.logspace(w1,w2,1000)
        s = 1j*w
        S = 1/(1+G*K)
        gain = np.abs(S(s))
        inv_gd = 1/Gd
        mag_i = np.abs(inv_gd(s))

        plt.figure('Rule 1')
        plt.loglog(w, gain, label = '|S|')
        plt.loglog(w, mag_i, ls = '--',label='|1/G_d |')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Magnitude')
        plt.legend(bbox_to_anchor=(0, 1.01, 1, 0), loc=3, ncol=3)
        plt.show()

    return valid1, wc, wd


def rule2(G, R, K, wr, message=False, plot=False, w1=-4, w2=2):
    """
    This is rule two of chapter five

    Calculates speed of response to track reference changes. Conditions require
    |S(jw)| <= 1/R

    Parameters
    ----------
    G : tf
        plant model

    R : real
        reference change

    K : tf
        control model

    wr : real
        reference frequency where tracking is required

    message : boolean
        show the rule message (optional)

    plot : boolean
        show the bode plot with constraints (optional)

    w1 : integer
        start frequency, 10^w1 (optional)

    w2 : integer
        end frequency, 10^w2 (optional)

    Returns
    -------
    invref : real
       1 / R

    """

    invref = 1/R

    if message:
        print('Rule 2:')
        print('Conditions requires |S(jw)| <= ', np.round(invref,2))

    if plot:
        w = np.logspace(w1,w2,1000)
        s = 1j*w
        S = 1/(1+G*K)
        gain = np.abs(S(s))

        plt.figure('Rule 2')
        plt.loglog(w, gain, label = '|S|')
        plt.loglog(w,  invref * np.ones(len(w)), ls = '--', label = '1/R')
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Magnitude')
        plt.legend(bbox_to_anchor=(0, 1.01, 1, 0), loc=3, ncol=3)
        plt.show()

    return invref


def rule3(G, Gd, message=False, w1=-4, w2=2):
    """
    This is rule three of chapter five

    Calculates input constraints arising from disturbance rejection

    Acceptable control conditions require |G(jw)| > |Gd(jw)| - 1'
    at frequencies where |Gd(jw) > 1|'

    Perfect control conditions require |G(jw)| > |Gd(jw)|'

    Parameters
    ----------
    G : tf
        plant model

    Gd : tf
        plant distrubance model

    message : boolean
        show the rule message (optional)

    w1 : integer
        start frequency, 10^w1 (optional)

    w2 : integer
        end frequency, 10^w2 (optional)

    """

    w = np.logspace(w1, w2, 1000)
    s = 1j * w

    mag_g = np.abs(G(s))
    mag_gd = np.abs(Gd(s))

    if message:
        print('Rule 3:')
        print('Acceptable control conditions require |G(jw)| > |Gd(jw)| - 1 at frequencies where |Gd(jw) > 1|')
        print('Perfect control conditions require |G(jw)| > |Gd(jw)|')

    plt.figure('Rule 3')
    plt.subplot(211)
    plt.title('Acceptable control')
    plt.loglog(w, mag_g, label = '|G|')
    plt.loglog(w, mag_gd - 1, label = '|Gd - 1|', ls = '--')
    plt.loglog(w,  1 * np.ones(len(w)))
    plt.legend(loc = 3)
    plt.grid()
    plt.ylabel('Magnitude')

    plt.subplot(212)
    plt.title('Perfect control')
    plt.loglog(w, mag_g, label = '|G|')
    plt.loglog(w, mag_gd, label = '|Gd|', ls = '--')
    plt.legend(loc = 3)
    plt.grid()
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Magnitude')
    plt.show()


def rule4(G, R, wr, message=False, w1=-4, w2=2):
    """
    This is rule four of chapter five

    Calculates input constraints arising from setpoints

    Conditions require |G(jw)| > R - 1 up to frequency wr

    Parameters
    ----------
    G : tf
        plant model

    R : real
        reference change

    wr : real
        reference frequency where tracking is required

    w1 : integer
        start frequency, 10^w1 (optional)

    w2 : integer
        end frequency, 10^w2 (optional)

    """

    w = np.logspace(w1, w2, 1000)
    s = 1j * w

    mag_g = np.abs(G(s))
    mag_rr = (R - 1)*np.ones(len(w))

    if message:
        print('Rule 4:')
        print('To avoid input saturation when setpoints change, we require |G(jw)| > R - 1')

    plt.figure('Rule 4')
    plt.loglog(w, mag_g, label='|G|')
    plt.loglog(w,  mag_rr, ls = '--', label='R - 1')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Magnitude')
    plt.legend(bbox_to_anchor=(0, 1.01, 1, 0), loc=3, ncol=3)
    plt.show()
    #df.setup_plot(['|G|', 'R - 1', '$w_r$'], wr, mag_g)


def rule5(G, Gm=1, message=False):
    """
    This is rule five of chapter five

    Calculates constraints for time delay, wc < 1 / theta

    Parameters
    ----------
    G : tf
        plant model

    Gm : tf
        measurement model

    message : boolean
        show the rule message (optional)

    Returns
    -------
    valid5 : boolean
        value if rule conditions was met

    wtd: real
        time delay frequency
    """

    GGm = G * Gm
    TimeDelay = GGm.deadtime
    _,_,wc,_ = margins(GGm)

    valid5 = False
    if TimeDelay == 0:
        wtd = 0
    else:
        wtd = 1 / TimeDelay
        valid5 = wc < wtd

    if message:
        print('Rule 5:')
        if TimeDelay == 0:
            print("There isn't any deadtime in the system")
        if valid5:
            print('wc < 1 / theta :', np.round(wc,2) , '<' , np.round(wtd,2))
        else:
            print('wc > 1 / theta :', np.round(wc,2) , '>' , np.round(wtd,2))

    return valid5, wtd


def rule6(G, Gm, message=False):
    """
    This is rule six of chapter five

    Calculates if tight control at low frequencies with RHP-zeros is possible

    Parameters
    ----------
    G : tf
        plant model

    Gm : tf
        measurement model

    message : boolean
        show the rule message (optional)

    Returns
    -------
    valid6 : boolean value if rule conditions was met

    wc : crossover frequency where | G(jwc) | = 1

    wd : crossover frequency where | Gd(jwd) | = 1

    """

    GGm = G * Gm
    zeros = GGm.zeros()

    _,_,wc,_ = margins(GGm)

    wz = 0
    if len(zeros) > 0:
        if np.imag(np.min(zeros)) == 0:
            # If the roots aren't imaginary.
            # Looking for the minimum values of the zeros => 
            # results in the tightest control.
            wz = (np.min(np.abs(zeros)))/2
            valid6 = wc < 0.86*np.abs(wz)
        else:
            wz = 0.86*np.abs(np.min(zeros))
            valid6 = wc < wz/2
    else: valid6 = False

    if message:
        print('Rule 6:')
        if wz != 0:
            print('These are the roots of the transfer function matrix GGm', zeros)
        if valid6:
            print('The critical frequency of S for the system to be controllable is', wz)
        else: print('No zeros in the system to evaluate')
    return valid6, wz


def rule7(G, Gm, message=False):
    """
    This is rule one of chapter five

    Calculates the phase lag constraints

    Parameters
    ----------
    G : tf
        plant model

    Gm : tf
        measurement model

    message : boolean
        show the rule message (optional)

    Returns
    -------
    valid1 : boolean
        value if rule conditions was met

    wc : real
        crossover frequency where | G(jwc) | = 1

    wd : real
        crossover frequency where | Gd(jwd) | = 1

    """
    # Rule 7 determining the phase of GGm at -180 deg.
    # This is solved visually from a plot.

    GGm = G*Gm
    _, _, wc, wu = margins(GGm)

    valid7 = wc < wu

    if message:
        print('Rule 7:')
        if valid7:
            print('wc < wu :' , wc , '<' , wu)
        else:
            print('wc > wu :' , wc , '>' , wu)

    return valid7, wu


def rule8(G, message=False):
    """
    This is rule one of chapter five

    This function determines if the plant is open-loop stable at its poles

    Parameters
    ----------
    G : tf
        plant model

    message : boolean
        show the rule message (optional)

    Returns
    -------
    valid1 : boolean
        value if rule conditions was met

    wc : real
        crossover frequency where | G(jwc) | = 1
    """
    #Rule 8 for critical frequency min value due to poles

    poles = G.poles()
    _,_,wc,_ = margins(G)

    wp = 0
    if np.max(poles) < 0:
        wp = 2*np.max(np.abs(poles))
        valid8 = wc > wp
    else: valid8 = False

    if message:
        print('Rule 8:')
        if valid8:
            print('wc > 2p :', wc , '>' , wp)
        else:
            print('wc < 2p :', wc , '<' , wp)

    return valid8, wp


def allSISOrules(G, deadtime, Gd, K, R, wr, Gm):

    _, wc, wd = rule1(G, Gd, K, True, True)
    rule2(G, R, K, 50, True, True)
    rule3(G, Gd, True)
    rule4(G, R, wr, True)
    _, wtd = rule5(G, Gm, True)
    _, wz = rule6(G, Gm, True)
    _, wu = rule7(G, Gm, True)
    _, wp = rule8(G, True)
    
    w = np.logspace(-4, 2, 1000)
    s = 1j*w
    L = G*K/(1+G*K)
    plt.figure('Combined SISO Controllability Rules')
    plt.loglog(w, np.abs(G(s)), label = '|G|')
    plt.loglog(w, np.abs(Gd(s)), label = '|Gd|')
    plt.loglog(w, np.abs(L(s)), label = '|L|')
    plt.axvline(wd, ls='--', color = 'k', label='$\omega_d$')
    plt.axvline(wc, ls='--', color = 'k', label='$\omega_c$')
    plt.scatter(wp, 1, color='blue', label='$2p$')
    plt.scatter(wz, 1, color='lime', label='$z/2$')
    plt.scatter(wu, 1, color='magenta', label='$\omega_u$')
    plt.scatter(wtd, 1, color='red', label='$1/ \theta$')
    plt.legend()
    plt.xlim([10**-4,10**4])
    plt.ylim([10**-1,10**3])
    plt.show()

if __name__ == '__main__': # only executed when called directly, not executed when imported

    s = tf([1, 0], 1)

    # Example plant based on Example 2.9 and Example 2.16
    G = (s + 200)/((10*s + 1)*(0.05*s + 1)**2)
    deadtime = 0.002
    Gd = 33/(10 * s + 1)
    K = 0.4*((s + 2)/s)*(0.075*s + 1)
    R = 3.0
    wr = 10
    Gm = 1 #Measurement model

    allSISOrules(G, deadtime, Gd, K, R, wr, Gm)
