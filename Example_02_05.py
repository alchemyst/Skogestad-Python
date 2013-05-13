import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from utils import phase, feedback

w = np.logspace(-1, 2, 1000)
s = w*1j
G = 4 /((s-1)*(0.02*s+1)**2)
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


# Calculate the frequency at which |L(jw)| = 1
wc = w[np.flatnonzero(np.abs(L) < 1)[0]]

# Calculate the frequency at which Angle[L(jw)] = -180
def Lu_180(w):
    s = w*1j
    G = 4 /((s-1)*(0.02*s+1)**2)
    Kc = 1.25
    tau1 = 1.5
    K = Kc*(1+1/(tau1*s))
    L = G*K
    return np.angle(L) + np.pi
w180 = sc.optimize.fsolve(Lu_180, 0.1)

# Calculate the phase margin
PM = Lu_180(wc)
print "PM:", np.round(PM*180/np.pi, 1), "deg or", np.round(PM, 2), "rad"

plt.show()
