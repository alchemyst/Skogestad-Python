# -*- coding: utf-8 -*-
"""
Created on Mon Jun 03 18:50:50 2013

@author: Irshad
"""

#Figure 5.22
import numpy as np
import matplotlib.pyplot as plt

th = 1000
kd = 2.5 * 10**6
w = np.logspace(-4, 5, 1000)
s = 1j*w


def G(s):
    return kd/(1 + th * s)


def Gd(s):
    return (-2 * kd)/(1 + th * s)

freqrespG = map(G, s)
freqrespGd = map(Gd, s)

plt.figure(1)
plt.title('|G| and |Gd| Value over Frequency')

plt.xlabel(r'Frequency [rad/s]', fontsize=14)
plt.ylabel(r'Magnitude', fontsize=15)
plt.loglog(w, np.abs(freqrespG))
plt.loglog(w, np.abs(freqrespGd))
plt.loglog(w, (np.ones(len(w))), '-.')
plt.vlines(2500, 10**(-2), 1, color='m', linestyle='dashed')
plt.grid(b=None, which='both', axis='both')
plt.legend(('G', 'Gd', 'Gain Value of 1', 'wd'), loc=1)
plt.show()
