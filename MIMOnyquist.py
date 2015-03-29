"""
Nyquist plot for a MIMO system.

Adapt the already specified transfer funtion matrix to match your system. 
"""

import numpy as np
from utilsplot import mino_nyquist_plot, plt

K = np.array([[1., 2.],
              [3., 4.]])
t1 = np.array([[5., 5.],
               [5., 5.]])
t2 = np.array([[5., 6.],
               [7., 8.]]) 
#Controller
Kc = np.array([[0.1, 0.], 
               [0., 0.1]])*6


def G(s):
    return(K*np.exp(-t1*s)/(t2*s + 1))


def L(s):
    return(Kc*G(s))
    
mino_nyquist_plot(L, 2, -3, 3)
plt.show()
