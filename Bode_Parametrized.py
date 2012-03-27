import cmath
import numpy as np
import scipy as sc
import scipy.signal as scs
import matplotlib.pyplot as plt
from utils import phase


def G(f, w):
    """system equations in here (laplace domain)
    only for SISO problems for now"""
    s = w*1j

    #Evaluate the model for the current wj.
    G = eval(f)

    return G

def Bode_Parametrized(f="8/((s+8)*(s+1))", startBaseFreq=-5, endBaseFreq=37):
    """Give the Bode plot for a given plant f in string format (e.g. 8/((s+8)*(s+1))).
       If no argument is provided the example is the default value of f.
       The starting frequency and ending frequencies for the plot are provided as minFreq and maxFreq."""

    def mod(x):
        """to give the function to calculate |G(jw)| = 1"""
        return np.abs(G(f, x)) - 1

    # how to calculate the frequency at which |G(jw)| = 1
    wc = sc.optimize.fsolve(mod, 0.1)

    def arg(w):
        """function to calculate the phase angle at -180 deg"""
        return np.angle(G(f, w)) + np.pi

    # where the frequency is calculated where arg G(jw) = -180 deg
    w_180 = sc.optimize.fsolve(arg, -1)

    # Plotting of gain Bode plot
    w = np.logspace(startBaseFreq, endBaseFreq, num=1000)

    #Set up the plot
    fig = plt.figure()
    plt.subplot(211)
    plt.loglog(w, np.abs(G(f, w)))
    #Plot the line for the unit gain
    plt.loglog(w, np.ones(1000), 'r--')
    plt.ylabel("Magnitude")
    plt.title("Figure 2.2: Frequency response (Bode plots) of G(s) = " + f + "\n ", fontsize=12)  # \n for a newline so that it doesn't hog the top of the graph.
    #Set the axis to the w range - adds in some x-space otherwise, sucks...
    plt.xlim(10.0**startBaseFreq, 10.0**endBaseFreq)

    # Plotting of phase Bode plot
    plt.subplot(212)
    phaseangle = phase(G(f, w), deg=True)
    plt.semilogx(w, phaseangle)
    plt.semilogx(w, -180 * np.ones(1000), 'r--')
    #Set the axis to the w range - adds in some x-space otherwise, sucks...
    plt.xlim(10.0**startBaseFreq, 10.0**endBaseFreq)
    plt.ylabel("Phase")
    plt.xlabel("Frequency [rad/s]")

    plt.show()

    return fig
