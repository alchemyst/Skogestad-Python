# -*- coding: utf-8 -*-
"""
Created on Mon Jun 03 23:16:52 2013

@author: Irshad
"""

#Figure 7.8
from numpy import exp, logspace, abs, ones
import matplotlib.pyplot as plt

# Time delay
w = logspace(-2, 2, 1000)
s = 1j * w
thetap = 1
l_delay = 1 - exp(-thetap * s)
tau = 1
l_nolag = 1 - (1/(1 + tau * s))


def setup_bode(G, p, func_string, max_param,  w=None):
    plt.figure()
    plt.xlabel(r'Frequency [rad/s]', fontsize=14)
    plt.ylabel(r'Magnitude', fontsize=15)
    plt.loglog(w, abs(G))
    plt.loglog(w, (ones(len(w))), '-.')
    plt.vlines(1, 10**(p), 1, color='m', linestyle='dashed')
    plt.ylim(-2, 10)
    plt.grid(b=None, which='both', axis='both')
    plt.legend((func_string, '1', max_param), loc=3)

func = [[l_delay, -2, 'time delay', r'$\theta_{max}$'],
        [l_nolag, -3, 'First order lag', r'$1/\tau_{max}$']]

for function, power, func_string, max_param in func:
    setup_bode(function, power, func_string, max_param, w)

plt.show()

