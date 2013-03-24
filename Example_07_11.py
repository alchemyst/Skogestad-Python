import numpy as np
import matplotlib.pyplot as plt
from utils import tf

WP = tf([0.25, 0.1], [1, 0])
WU = tf([1, 0], [1, 1])
GK1 = tf([0, 0.5], [1, 0])
GK2 = tf([0, -0.5, 0.5], [1, 1, 0])

w = np.logspace(-2, 2, 500)
s = 1j*w
ru = np.linspace(0, 1, 10)

"""Solution to part a,b is on pg 284, only part c is illustrateed here"""
plt.loglog(w, np.abs(1/(1 + GK1(s))), label='$S_1$')
plt.loglog(w, np.abs(1/(1 + GK2(s))), label='$S_2$')

for i in range(np.size(ru)):
    if i == 1:
        plt.loglog(w, 1/(np.abs(WP(s)) + np.abs(ru[i]*WU(s))), 'k--', label='varying $r_u$')
    else:
        plt.loglog(w, 1/(np.abs(WP(s)) + np.abs(ru[i]*WU(s))), 'k--')

plt.loglog(w, 1/(np.abs(WP(s)) + np.abs(0.75*WU(s))), 'r', label='$r_u$ = 0.75')
plt.legend(loc=2)
plt.title(r'Figure 7.19')
plt.xlabel(r'Frequency [rad/s]', fontsize=14)
plt.ylabel(r'Magnitude', fontsize=15)
plt.show()
