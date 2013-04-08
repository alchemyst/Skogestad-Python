# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:10:40 2013

@author: Irshad
"""

import numpy
import scipy
import scipy.signal
import matplotlib.pyplot as plt

w = numpy.logspace(-3, 2, 2000)
s = w*1j
G = (1-s)/(s+1)

for kc in [0.1, 0.5, 0.9]:
    k1 = -1*kc*(s)/((1+(0.02*s))*(1+(0.05*s)))
    L = G*k1
    S = 1/(1+L)
    plt.loglog(w, abs(S))
plt.show()

print s
