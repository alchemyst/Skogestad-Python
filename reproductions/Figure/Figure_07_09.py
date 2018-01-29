import numpy as np
import matplotlib.pyplot as plt
from math import pi
from utils import tf

def LI(rk, tmax, w):
    L = np.zeros(np.size(w))
    for i in range(np.size(w)):
        if w[i-1] < pi/tmax:
            L[i-1] = np.sqrt(rk**2 + 2*(1+rk)*(1-np.cos(tmax*w[i-1])))
        else:
            L[i-1] = 2 + rk
    return L

rk = 0.2
tmax = 1

WI = tf([(1 + rk*0.5)*tmax, rk], [tmax*0.5, 1])
WI_improved = WI*tf([(tmax/2.363)**2, 2*0.838*(tmax/2.363), 1], [(tmax/2.363)**2, 2*0.685*(tmax/2.363), 1])

w = np.logspace(-2, 2, 500)
s = 1j*w

plt.loglog(w, LI(rk, tmax, w), 'r', label='$l_I$')
plt.loglog(w, np.abs(WI(s)), 'b--', label='$\omega_I$')
plt.loglog(w, np.abs(WI_improved(s)), color = 'lime', label='$improved$ $\omega_I$')
plt.legend()
plt.title(r'Figure 7.9')
plt.xlabel(r'Frequency [rad/s]', fontsize=12)
plt.ylabel(r'Magnitude', fontsize=12)
plt.show()
