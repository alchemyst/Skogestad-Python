'''
Created on 24 Mar 2013

@author: St Elmo Wilken
'''

import control as cn
import matplotlib.pyplot as plt
import numpy as np
import control_add_on as cadd
import siso_controllability as scont


"""
Scaling: x* refers to the unscaled perturbation variable
q = q*/1
T_0 = T_0*/10
T = T*/10
"""

G = cn.tf([0.8],[60, 1])*cn.tf([1],[12, 1]) #this is the only tf which changes due to scaling => equates to *0.1
Gd = cn.tf([20*0.6, 0.6],[60, 1])*cn.tf([1],[12,1])
# remember the dead time in measurement which is 3 seconds
freqs = np.arange(0.001, 1, 0.001)
plt.figure(1)
scont.rule_three_four(G, Gd, R = 2, perfect = True, freq = freqs)

plt.show()
"""
Performance:
The plant will perform well to input (R) and disturbance tracking as it
is self regulating i.e. |Gd| < 1 for all frequencies.

Controllability:
It is self regulating so... this is awkward...
"""