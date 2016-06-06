from __future__ import print_function
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def G(w):
    """system equations in here (laplace domian)
    only for SISO problems for now"""
    s = w*1j

    # enter what ever system under inspections, transfer function in here
    G = 8/((s+0.01)*(s+3)*(s+2))

    return G


def Nyquist(w_start, w_end):
    """w_start and w_end is the number that would be 10**(number)"""
    def mod(w):
        return np.abs(G(w))-1

    w_start_n = fsolve(mod, 0.001)
    print(w_start_n)

    plt.plot(np.real(G(w_start_n)), np.imag(G(w_start_n)), 'rD')
    w_start = np.log(w_start_n)

    w = np.logspace(w_start, w_end, 100)
    x = np.real(G(w))
    y = np.imag(G(w))
    plt.plot(x, y, 'b+')
    plt.xlabel('Re G(wj)')
    plt.ylabel('Im G(wj)')

    # plotting a unit circle
    x = np.linspace(-1, 1, 200)

    y_upper = np.sqrt(1 - x**2)
    y_down = -1*np.sqrt(1 - x**2)
    plt.plot(x, y_upper, 'r-', x, y_down, 'r-')

    plt.show()

    print("finished")

Nyquist(-3, 5)
