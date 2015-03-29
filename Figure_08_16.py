# -*- coding: utf-8 -*-
"""
Created on Tue Jun 04 00:41:19 2013

@author: Irshad
"""

#Figure 8.16
import numpy as np
import matplotlib.pyplot as plt

w = np.logspace(-3, 2, 1000)
s = 1j*w

Wi = (s + 0.2)/(0.5 * s + 1)
Wp = (s/2 + 0.05)/s

plt.figure(1)
plt.title('Weight Values over Frequency')

plt.xlabel(r'Frequency [rad/s]', fontsize=14)
plt.ylabel(r'Magnitude', fontsize=15)
plt.loglog(w, np.abs(Wi))
plt.loglog(w, np.abs(Wp))
plt.loglog(w, (np.ones(len(w))), '-.')
plt.grid(b=None, which='both', axis='both')
plt.legend(('Wi', 'Wp', 'Gain Value of 1'), loc=1)
plt.show()
