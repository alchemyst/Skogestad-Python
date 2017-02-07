# -*- coding: utf-8 -*-
"""
Created on Sat May 31 15:59:13 2014

@author: pedawson

Skogestad Exercise 2.4
"""

import numpy as np
import matplotlib.pyplot as plt


M = 2.
A = 0.1
wB = 1
n = 2.
w = np.logspace(-2,2,1000)
s = w*1j
Wp_2_105 = (s/M + wB)/(s + wB*A)
Wp_2_106 = ((s/(M**(1/n)) + wB)**n)/((s + wB*A**(1/n))**n)
plt.figure('Performance Weights')
plt.clf()
plt.loglog(w, 1/Wp_2_105, 'k-', label='$W_p2.105$')
plt.loglog(w, 1/Wp_2_106, 'b-', label='$W_p2.106$')
plt.legend(loc='upper left', fontsize=10, ncol=1)
plt.axhline(1, ls=':', color='red', lw=2)
plt.text(0.012, 1.1, 'Mag = 1', color='red', fontsize=10)
plt.text(40, 2.6, 'Max |S|', color='green', fontsize=10)
plt.text(10, 2.1, 'Peak specification (M)', color='green', fontsize=10)
plt.text(0.011, 0.08, 'Max SS tracking error (A)', color='green', fontsize=10)
plt.axvline(wB, ls=':', color='red', lw=2)
plt.text(wB*1.1, 8, 'wB', color='red', fontsize=10)
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude')
plt.grid(True)
fig = plt.gcf()
BG = fig.patch
BG.set_facecolor('white')
plt.show()


