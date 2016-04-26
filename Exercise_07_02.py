# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:10:25 2015

@author: cronjej
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import doc_func as df

# Gprime = [[part, range of parameter, function, True if min and max values wanted, legend string]]
Gprime = [['a)', np.linspace(0,1,101), df.Gp_a, False, '$l_{I}$'],
          ['b)', np.linspace(0,1,101), df.Gp_b, False, '$l_{I}$'],
          ['c)', np.linspace(0,2,201), df.Gp_c, True, '$l_{I}(p_{min})$'],
          ['d)', np.linspace(0,2,201), df.Gp_d, True, r'$l_{I}(\tau_{min})$'],
          ['e)', np.linspace(0,2,201), df.Gp_e, True, '$l_{I}(\zeta_{max})$'],
          ['f)', np.linspace(0,20,21), df.Gp_f, True, '$l_{I}$'],
          ['g)', np.linspace(0,1,101), df.Gp_g, False, '$l_{I}$']]

df.plot_range(df.G, Gprime, df.wI, np.logspace(-3, 3, 200))
