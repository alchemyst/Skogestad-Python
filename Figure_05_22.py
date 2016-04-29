# -*- coding: utf-8 -*-
"""
Created on Mon Jun 03 18:50:50 2013

@author: Irshad
"""
from __future__ import division
#Figure 5.22
import numpy as np
import matplotlib.pyplot as plt
import doc_func as df

th = 1000
kd = 2.5 * 10**6
w = np.logspace(-4, 5, 1000)
s = 1j*w


def G(s):
    return kd/(1 + th * s)


def Gd(s):
    return (-2 * kd)/(1 + th * s)

freqrespG = [G(si) for si in s]
freqrespGd = [Gd(si) for si in s]

func_list = [[np.abs(freqrespG), '-'],
             [np.abs(freqrespGd), '-'],
             [np.ones(len(w)), 'r-.']]

plt.vlines(2500, 10**(-2), 1, color='m', linestyle='dashed')
df.setup_bode_plot('|G| and |Gd| Value over Frequency', w, func_list, legend=('G', 'Gd', 'Gain Value of 1', 'wd'))
