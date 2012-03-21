import numpy as np
import scipy as sc
import scipy.signal
import matplotlib.pyplot as plt
from utils import phase

w = np.logspace(-3, 1, 1000)
s = w*1j
kc = 1.0
G = 4/((s-1)*(0.02*s+1)**2)
L = kc*G

#magnitude and phase
plt.subplot(2, 1, 1)
plt.loglog(w, abs(L))
plt.subplot(2, 1, 2)
plt.semilogx(w, phase(L, deg=True))
plt.semilogx(w, (-180)*np.ones(len(w)))
plt.show()
