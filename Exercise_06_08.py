# -*- coding: utf-8 -*-
"""
Created on Sat Jun 01 22:29:33 2013

@author: Irshad
#Exercise6.8
#Analyse the Controllability of G(s)
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
w = np.logspace(-2, 2, 1000)
s = 1j*w


def G(s):
    return (1/(s**2 + 100)) * np.matrix([[(1/(0.01 * s + 1)), 1],
                                         [((s + 0.1)/(s + 1)), 1]])


def g11(s):
    return (1/(s**2 + 100)) * (1/(0.01 * s + 1))


def g12(s):
    return 1/(s**2 + 100)


def g21(s):
    return (1/(s**2 + 100)) * ((s + 0.1)/(s + 1))


def g22(s):
    return 1/(s**2 + 100)


def lambda11(s):
    return 1/(1 - ((g12(s) * g21(s))/(g11(s) * g22(s))))

#Lambda11=Lambda22
#Lambda12=Lambda21

freqresp = map(G, s)
l11 = np.array([lambda11(i) for i in s])
l21 = 1 - l11
sigmas = np.array([Sigma for U, Sigma, V in map(la.svd, freqresp)])
plt.figure(1)
plt.title('RGA Values over Frequency')
plt.xlabel(r'Frequency [rad/s]', fontsize=14)
plt.ylabel(r'Magnitude', fontsize=15)
plt.loglog(w, abs(l11))
plt.loglog(w, abs(l21))
plt.grid(b=None, which='both', axis='both')
plt.legend(('Lambda11/Lambda22', 'lambda12/Lambda21'), loc=2)
plt.figure(2)
plt.title('SVD Values over Frequency')
plt.xlabel(r'Frequency [rad/s]', fontsize=14)
plt.ylabel(r'Magnitude', fontsize=15)
plt.loglog(w, sigmas)
plt.grid(b=None, which='both', axis='both')
plt.show()
