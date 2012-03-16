import numpy as np
import scipy as sc
import scipy.signal 
import matplotlib.pyplot as plt
w = np.logspace(-3,1,1000)
s = w*1j
kc = 1.0
G = 4/((s-1)*(0.02*s+1)**2)
L = kc*G
def phase(L):
    return np.unwrap(np.arctan2(np.imag(L),np.real(L)))
#magnitude and phase
plt.subplot(2,1,1)
plt.loglog(w,abs(L))
plt.subplot(2,1,2)
plt.semilogx(w,(180/np.pi)*phase(L))
plt.semilogx(w,(-180)*np.ones(len(w)))
plt.show()