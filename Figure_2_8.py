import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scs
import scipy as sc
from utils import phase, Closed_loop


def Wu_180(w):
    s = w*1j
    G = 3*(-2*(s)+1)/((10*s+1)*(5*s+1))
    Wu = np.angle(G) + np.pi
    return Wu


def G_w(w):
    s = w*1j
    G = 3*(-2*(s)+1)/((10*s+1)*(5*s+1))
    mod = np.abs(G)
    return mod


def Zeigler_Nichols():
    wu = sc.optimize.fmin_bfgs(Wu_180, np.pi, fprime=Wu_180)
    mod = G_w(wu)
    Ku = 1/mod
    Pu = 2*np.pi/wu
    Kc = Ku/2.2
    Tc = np.abs(Pu/1.2)

    Kz = np.hstack([Kc*Tc, 1])
    Kp = np.hstack([Tc, 0])
    return Kc, Tc, Kz, Kp

# calculating the ultimate values of the controller constants
# Ziegler and Nichols controller tuning parameters
[Kc, Tc, Kz, Kp] = Zeigler_Nichols()
print Kc, Tc, Kz, Kp

Gz = [-6, 3]
Gp = [50, 15, 1]


[Z_cl_poly, P_cl_poly] = Closed_loop(Kz, Kp, Gz, Gp)

f = scs.lti(Z_cl_poly, P_cl_poly)
print Z_cl_poly, P_cl_poly

[t, y] = f.step()

plt.plot(t, y)
plt.show()
