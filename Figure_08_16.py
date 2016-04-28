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

Wi = (s + 0.2)/(0.5 * s + 1)
Wp = (s/2 + 0.05)/s

func_list = [[np.abs(Wi), '-', False],
             [np.abs(Wp), '-', False],
             [np.ones(len(w)), 'r-.', True]]

plot = plt.loglog
for function, linestyle, grid_bool in func_list:
    df.setup_bode_plot('Weight Values over Frequency', w, function, plot, grid_bool, linestyle)

plt.legend(('Wi', 'Wp', 'Gain Value of 1'), loc='best')

plt.show()
