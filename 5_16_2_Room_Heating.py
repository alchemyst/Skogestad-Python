"""
Created on 24 Mar 2013

@author: St Elmo Wilken
"""
import control as cn
import matplotlib.pyplot as plt
import numpy as np

import siso_controllability as scont

# There is a time delay from measurement of 100s.
# This implies that the upper bound for w_c is about 1/100 = 0.01 rad/s.

G = cn.tf([20], [1000, 1])
Gd = cn.tf([10], [1000, 1])
# from the solution
K = cn.tf([0.4], [1])*cn.tf([200, 1], [200, 0])*cn.tf([60, 1], [1])

plt.figure("Input Constraint Bode")
freqs = np.arange(0.00001, 0.1,
                  0.00001)  # this is the frequency range we are dealing with
scont.rule_three_four(G, Gd, R=3, freq=freqs, perfect=False)

# From Disturbance Bode you can see |G| > |Gd| always. This implies that
# there will not be any input constraints wrt disturbances. (Rule 3)
# I.e. your system will always be strong enough to counteract your disturbances.

# Using R = 3 you get an intersection of w_r = 0.0065 which implies that
# the plant will be able to track set point changes up to w_r. The required response time
# is 1/1000 = 0.001. Therefore w_r is better than the required response time.

plt.figure("Feedback Constraint Bode")
S = 1/(1 + G*K)
scont.rule_one_two(S, Gd, R=3, freq=freqs)

# From Feedback Constraint Bode you can see that the feedback loop of the
# system will be able to handle any disturbance input.

# However, at about w = 0.004 rad/s you can see that setpoints will not be tracked
# that is, the error will be more than one.

plt.figure("Dead Time Constraint on cross over frequency")
L = G*K
scont.dead_time_bound(L, Gd, 100, freq=freqs)

# From the output you can see that because the upper bound of w_c is so close to the lower bound of
# w_c you are going to have a problem.


plt.show()
