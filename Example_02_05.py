import numpy as np
import matplotlib.pyplot as plt
from utils import phase, feedback

w = np.logspace(-1, 2, 1000)
s = w*1j
G = 4/((s-1)*(0.02*s+1)**2)
Kc = 1.25
tau1 = 1.5
K = Kc*(1+1/(tau1*s))
L = K*G
S = feedback(1,L)
T = feedback(L,1)


#Bode magnitude and phase plot - Figure 2.15
plt.subplot(2, 1, 1)
plt.loglog(w, abs(L))
plt.loglog(w, abs(S))
plt.loglog(w, abs(T))
plt.ylabel("Magnitude")
plt.legend(["L","S","T"],
           bbox_to_anchor=(0, 1.01, 1, 0), loc=3, ncol=3)
plt.subplot(2, 1, 2)
plt.semilogx(w, phase(L, deg=True))
plt.semilogx(w, phase(S, deg=True))
plt.semilogx(w, phase(T, deg=True))
plt.semilogx(w, (-180)*np.ones(len(w)))
plt.ylabel("Phase")
plt.xlabel("Frequency [rad/s]")
plt.show()

