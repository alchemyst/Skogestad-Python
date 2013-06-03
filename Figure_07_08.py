# -*- coding: utf-8 -*-
"""
Created on Mon Jun 03 23:16:52 2013

@author: Irshad
"""

#Figure 7.8
import numpy as np
import matplotlib.pyplot as plt
# Time delay
w = np.logspace(-2, 2, 1000)
s = 1j * w
thetap = 1
l_delay = 1 - np.exp(-thetap * s)
tau = 1
l_nolag = 1 - (1/(1 + tau * s))

plt.title('Time Delay Value over Frequency')
plt.figure(1)
plt.xlabel(r'Frequency [rad/s]', fontsize=14)
plt.ylabel(r'Magnitude', fontsize=15)
plt.loglog(w, np.abs(l_delay))
plt.loglog(w, (np.ones(len(w))), '-.')
plt.vlines(1, 10**(-3), 1, color='m', linestyle='dashed')
plt.ylim(-2, 10)
plt.grid(b=None, which='both', axis='both')
plt.legend(('time delay', '1', 'theta_max'), loc=3)
plt.figure(2)
plt.xlabel(r'Frequency [rad/s]', fontsize=14)
plt.ylabel(r'Magnitude', fontsize=15)
plt.loglog(w, np.abs(l_nolag))
plt.loglog(w, (np.ones(len(w))), '-.')
plt.vlines(1, 10**(-2), 1, color='m', linestyle='dashed')
plt.ylim(-2, 10)
plt.grid(b=None, which='both', axis='both')
plt.legend(('First-order Lag', '1', '1/tau_max'), loc=1)
plt.show()
