import numpy as np
import scipy as sc
import scipy.signal as scs
import matplotlib.pyplot as plt
from utils import phase


def G(s):
    """system equations in here (laplace domian)
    only for SISO problems for now"""

    # enter what ever system under inspections, transfer function in here
    G = 8/((s+8)*(s+1))

    return G

# freq(G) returns a frequency response function given a laplace function
def freq(G):
    def Gw(w):
        return G(1j*w)
    return Gw

def margins(G):
    """ Calculate the gain margin and phase margin of a system.
    
    Input: G - a function of s
    Outputs:
       GM    Gain margin
       PM    Phase margin
       wc    Gain crossover frequency
       w_180 Phase Crossover frequency
    """
    
    Gw = freq(G) 
     
    def mod(x):
        """to give the function to calculate |G(jw)| = 1"""
        return np.abs(Gw(x)) - 1

    # how to calculate the freqeuncy at which |G(jw)| = 1
    wc = sc.optimize.fsolve(mod, 0.1)

    def arg(w):
        """function to calculate the phase angle at -180 deg"""
        return np.angle(Gw(w)) + np.pi

    # where the freqeuncy is calculated where arg G(jw) = -180 deg
    w_180 = sc.optimize.fsolve(arg, -1)

    PM = np.angle(Gw(wc), deg=True) + 180
    GM = 1/(np.abs(Gw(w_180)))

    return GM, PM, wc, w_180


def Bode(G):
    """give the Bode plot along with GM and PM"""

    GM, PM, wc, w_180 = margins(G)

    # plotting of Bode plot and with corresponding freqeuncies for PM and GM
    w = np.logspace(-5, np.log(w_180), 1000)
    s = 1j*w
    
    plt.subplot(211)
    gains = np.abs(G(s))
    plt.loglog(w, gains)
    plt.loglog(wc*np.ones(2), [np.max(gains), np.min(gains)])
    plt.text(w_180, np.average([np.max(gains), np.min(gains)]), '<G(jw) = -180 Deg')
    plt.loglog(w_180*np.ones(2), [np.max(gains), np.min(gains)])
    plt.loglog(w, 1*np.ones(len(w)))

    # argument of G
    plt.subplot(212)
    phaseangle = phase(G(s), deg=True)
    plt.semilogx(w, phaseangle)
    plt.semilogx(wc*np.ones(2), [np.max(phaseangle), np.min(phaseangle)])
    plt.semilogx(w_180*np.ones(2), [-180, 0])
    plt.show()

    return GM, PM


[GM, PM] = Bode(G)

print GM, PM
