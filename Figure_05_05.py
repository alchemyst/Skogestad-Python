# -*- coding: utf-8 -*-
"""
Created on Mon Jun 03 19:36:05 2013

@author: Irshad
"""

"This code plots figure 5.5"

import numpy as np
import matplotlib.pyplot as plt

w = np.linspace(0, 6, 1000)
s = 1j * w


def L1(s):
    return 2/(s * (s + 1))


def L2(s):
    return L1(s) * ((5 - s)/(s + 5))


def S1(s):
    return 1/(L1(s) + 1)


def S2(s):
    return 1/(L2(s) + 1)

freqrespS1 = map(S1, s)
freqrespS2 = map(S2, s)

plt.figure(1)
plt.title('|S1| and |S2| Value over Frequency')

plt.xlabel(r'Frequency [rad/s]', fontsize=14)
plt.ylabel(r'Magnitude', fontsize=15)
plt.semilogy(w, np.abs(freqrespS1))
plt.semilogy(w, np.abs(freqrespS2))
plt.semilogy(w, (np.ones(len(w))), '-.')
plt.grid(b=None, which='both', axis='both')
plt.legend(('S1', 'S2', 'Gain Value of 1'), loc=1)
plt.show()
