import numpy as np
import matplotlib.pyplot as plt

w = np.logspace(-2, 1, 1000)
s = w*1j

plt.figure('Figure 5.4')
Ks = [0.1, 0.5, 1.0, 2.0]
for k in Ks:
    L = k/s * (2 - s)/(2 + s)
    T = L/(1+L)
    S = 1 - T
    plt.loglog(w, abs(S))
plt.legend(["k = %1.1f" % K for K in Ks], loc=2)
plt.xlabel('Frequency')        
plt.ylabel('Magnitude $|S|$')
plt.show()
