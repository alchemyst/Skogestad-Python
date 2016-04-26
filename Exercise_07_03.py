# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:10:25 2015

@author: cronjej
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import doc_func as df

Gprime2 = [['a)', np.linspace(0, 2, 201), df.Gp_a, False, '$l_{I}$'],
           ['b)', np.linspace(0, 2, 201), df.Gp_b, False, '$l_{I}$'],
           ['c)', np.linspace(0, 2, 201), df.Gp_c, True, '$l_{I}(p_{min})$'],
           ['d)', np.linspace(0, 3, 301), df.Gp_d, True, r'$l_{I}(\tau_{min})$'],
           ['e)', np.linspace(0, 8, 801), df.Gp_e, True, '$l_{I}(\zeta_{max})$'],
           ['f)', np.linspace(100, 150, 51), df.Gp_f, True, '$l_{I}$'],
           ['g)', np.linspace(0, 1, 101), df.Gp_g, False, '$l_{I}$']]


def wI(s):
    return (s + 0.3)/((1/3)*s + 1)

df.plot_range(df.G, Gprime2, wI, np.logspace(-3, 3, 200))
