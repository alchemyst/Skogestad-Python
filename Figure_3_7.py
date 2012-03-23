import numpy as np
import numpy.linalg as la
from matplotlib.pyplot import *

"""def the function for Figure 3.7)a the distilation process"""
def G1(s):
    return 1/(75*s+1)*np.array([[87.8, -86.4], [108.2, -109.6]])

"""def the function for Figure 3.7)b the spinning satelite"""
def G2(s):
    return 1/(s**2+10**2)*np.array([[s-10**2, 10*(s+1)], [-10*(s+1), s-10**2]])

"""Function to determine singular values """
def SVD(G, s):
    freqresp = map(G, s)
    sigmas = np.matrix([Sigma for U, Sigma, V in map(la.svd, freqresp)])
    return sigmas

"""Plotting of Figure 3.7)a and 3.7)b"""
subplot(1, 2, 1)
w = np.logspace(-4, 1, 1000)
loglog(w, SVD(G1, 1j*w))
xlabel(r'Frequency [rad/s]', fontsize=14)
ylabel(r'Magnitude', fontsize=15)
title('Distillation process 3.7(a)')
text(0.001, 220, r'$\bar \sigma$(G)', fontsize=15)
text(0.001, 2, r'$\frac{\sigma}{}$(G)', fontsize=20)
subplot(1, 2, 2)
w = np.logspace(-2, 2, 1000)
loglog(w, SVD(G2, 1j*w))
xlabel(r'Frequency [rad/s]', fontsize=14)
title('Spinning satellite 3.7(b)')
text(2, 20, r'$\bar \sigma$(G)', fontsize=15)
text(1, 0.2, r'$\frac{\sigma}{}$(G)', fontsize=20)
show()
