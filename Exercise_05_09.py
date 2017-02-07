'''
Created on Jun 6, 2013

@author: FENNERK
'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


def omega_p(s, k, M):
    return ((1000*s/k+1/M)*(s/(M*k)+1))/((10*s/k+1)*(100*s/k+1))

#plotting M = 2 and w_B = 1
ws = np.logspace(-4, 4, 1000)
ks = [0.1, 0.5, 1.0, 10.0, 100.0, 1000.0, 2000.0, 10000.0]

omegas = []

for w in ws:
    s = 1j*w
    omegas.append(1/np.abs(omega_p(s, 1.0, 2.0)))

plt.loglog(ws, omegas)
plt.xlabel("Freq")
plt.ylabel("$1/W_p$")
plt.show()


for k in ks:
    print("For k = ", k, ", Wp(z) = ", omega_p(1.0, k, 2.0))
 
