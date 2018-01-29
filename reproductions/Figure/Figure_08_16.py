# -*- coding: utf-8 -*-
"""
Created on Tue Jun 04 00:41:19 2013

@author: Irshad
"""

#Figure 8.16
import numpy as np
import matplotlib.pyplot as plt
import doc_func as df

w = np.logspace(-3, 2, 1000)
s = 1j*w

Wi = (s + 0.2)/(0.5*s + 1)
Wp = (s/2 + 0.05)/s

func_list = [[np.abs(Wi), '-'],
             [np.abs(Wp), '-'],
             [np.ones(len(w)), 'r-.']]

df.setup_bode_plot('Weight Values over Frequency', w, func_list, legend=('Wi', 'Wp', 'Gain Value of 1'))
