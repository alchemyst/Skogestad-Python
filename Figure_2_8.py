import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scs
import scipy as sc

# TODO: Don't duplicate functions
def Closed_loop(Kz, Kp, Gz, Gp):
    """this function return s the polynomial constants of the closed loop transfer function's numerator and denominator

    Kz is the polynomial constants in the numerator
    Kp is the polynomial constants in the denominator
    Gz is the polynomial constants in the numerator
    Gp is the polynomial constants in the denominator"""

    # calculating the product of the two polynomials in the numerator and denominator of transfer function GK
    Z_GK = np.polymul(Kz, Gz)
    P_GK = np.polymul(Kp, Gp)
    # calculating the polynomial of closed loop function T = (GK/1+GK)
    Zeros_poly = Z_GK
    Poles_poly = np.polyadd(Z_GK, P_GK)
    return Zeros_poly, Poles_poly


def Wu_180(w):
    s = w*1j
    G = 3*(-2*(s)+1)/((10*s+1)*(5*s+1))
    Wu = (np.arctan2(np.imag(G), np.real(G)))+np.pi
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

    Kc = np.number([Kc])
    Tc = np.number([Tc])
    Kz = [Kc*Tc, 1]
    Kp = [Tc, 0]
    return Kc, Tc

# calculating the ultimate values of the controller constants
# Ziegler and Nichols controller tuning parameters
[Kc, Tc] = Zeigler_Nichols()
print Kc, Tc

print Kz, Kp
Gz = [-6, 3]
Gp = [50, 15, 1]


[Z_cl_poly, P_cl_poly] = Closed_loop(Kz, Kp, Gz, Gp)

f = scs.lti(Z_cl_poly, P_cl_poly)
print Z_cl_poly, P_cl_poly

[t, y] = f.step()

plt.plot(t, y)
plt.show()
