import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

w = np.logspace(-2, 2, 1000)
s = w*1j
for kc in [0.2, 0.5, 0.8, 1]:
    G = (-s+1)/(s+1)
    k1 = kc*((s+1)/s)*(1/(0.05*s+1))
    L = k1*G
    T = L/(1+L)
    S = 1-T
    plt.loglog(w, abs(S))

plt.show()
