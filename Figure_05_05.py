# -*- coding: utf-8 -*-
"""
Created on Mon Jun 03 19:36:05 2013

@author: Irshad
"""

"This code plots figure 5.5"

import numpy as np
import matplotlib.pyplot as plt
import doc_func as df

w = np.linspace(0.001, 6, 1000)
s = 1j * w


def L1(s):
    return 2/(s * (s + 1))


def L2(s):
    return L1(s) * ((5 - s)/(s + 5))


def S1(s):
    return 1/(L1(s) + 1)


def S2(s):
    return 1/(L2(s) + 1)

freqrespS1 = np.abs(list(map(S1, s)))
freqrespS2 = np.abs(list(map(S2, s)))

func_list = [[freqrespS1, '-', False],
             [freqrespS2, '-', False],
             [np.ones(len(w)), 'r-.', True]]

plot = plt.semilogy
for func, lstyle, grid in func_list:
    df.setup_bode_plot('|S1| and |S2| Value over Frequency', w, func, plot, grid, lstyle)
plt.legend(('S1', 'S2', 'Gain Value of 1'),loc='best')
plt.show()
