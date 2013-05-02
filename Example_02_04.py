import matplotlib.pyplot as plt
import scipy.signal as scs
import numpy as np
from utils import Closed_loop, feedback, phase
import scipy as sc


# Ziegeler and Nicholas tuning rule
# Part of Excercise 2.1

def Wu_180_a(w):
    # Methode 1 to calculate the frequency where arg G(jw) = -180 deg
    return np.arctan(-2*w) - np.arctan(5*w) - np.arctan(10*w) - np.pi
wu_a = sc.optimize.fsolve(Wu_180_a, 0)
print 'Methode 1 wu:', wu_a


def Wu_180(w):
    s = w*1j
    G = 3*(-2*(s)+1)/((10*s+1)*(5*s+1))
    wu = np.angle(G) + np.pi
    return wu


def G_w(w):
    s = w*1j
    G = 3*(-2*(s)+1)/((10*s+1)*(5*s+1))
    mod = np.abs(G)
    return mod


def Zeigler_Nichols():
    # Methode 2 to calculate the frequency where arg G(jw) = -180 deg
    wu = sc.optimize.fmin_bfgs(Wu_180, np.pi, fprime=Wu_180)
    print 'Methode 2 wu:', wu
    mod = G_w(wu)
    Ku = 1/mod
    Pu = np.abs(2*np.pi/wu)
    print 'Ku:', Ku
    print 'Pu:', Pu
    Kc = Ku/2.2
    Tauc = Pu/1.2

    Kz = np.hstack([Kc*Tauc, 1])
    Kp = np.hstack([Tauc, 0])
    return Kc, Tauc, Kz, Kp

# calculating the ultimate values of the controller constants
# Ziegler and Nichols controller tuning parameters
[Kc, Tauc, Kz, Kp] = Zeigler_Nichols()
print Kc, Tauc, Kz, Kp

Gz = [-6, 3]
Gp = [50, 15, 1]

[Z_cl_poly, P_cl_poly] = Closed_loop(Kz, Kp, Gz, Gp)

f = scs.lti(Z_cl_poly, P_cl_poly)
print Z_cl_poly, P_cl_poly

[t, y] = f.step()

plt.title('Figure 2.8')
plt.xlabel('Time [s]')
plt.plot(t, y)
plt.figure()


# Bode magnitude and phase plot - Figure 2.14
w = np.logspace(-2, 1, 1000)
s = w*1j
K = Kc*(1+1/(Tauc*s))
G = 3*(-2*(s)+1)/((10*s+1)*(5*s+1))
L = G*K
S = feedback(1, L)
T = feedback(L, 1)

plt.subplot(2, 1, 1)
plt.loglog(w, abs(L))
plt.loglog(w, abs(S))
plt.loglog(w, abs(T))
plt.ylabel("Magnitude")
plt.legend(["L", "S", "T"],
           bbox_to_anchor=(0, 1.01, 1, 0), loc=3, ncol=3)
plt.subplot(2, 1, 2)
plt.semilogx(w, phase(L, deg=True))
plt.semilogx(w, phase(S, deg=True))
plt.semilogx(w, phase(T, deg=True))
plt.ylabel("Phase")
plt.xlabel("Frequency [rad/s]")


# Calculate the frequency at which |L(jw)| = 1
wc = w[np.flatnonzero(np.abs(L) < 1)[0]]

# Calculate the frequency at which Angle[L(jw)] = -180


def Lu_180(w):
    s = w*1j
    G = 3*(-2*(s)+1)/((10*s+1)*(5*s+1))
    K = Kc*(1+1/(Tauc*s))
    L = G*K
    return np.angle(L) + np.pi
w180 = sc.optimize.fsolve(Lu_180, 0.1)
# Calculate the gain margin


def L_w(w):
    s = w*1j
    G = 3*(-2*(s)+1)/((10*s+1)*(5*s+1))
    K = Kc*(1+1/(Tauc*s))
    L = G*K
    return np.abs(L)
GM = 1/L_w(w180)

# Calculate the phase margin
PM = Lu_180(wc)
print "GM:", np.round(GM, 2)
print "PM:", np.round(PM*180/np.pi, 1), "deg or", np.round(PM, 2), "rad"

# Calculate the frequency at which |S(jw)| = 0.707
wb = w[np.flatnonzero(np.abs(S) < (1/np.sqrt(2)))[0]]
# Calculate the frequency at which |T(jw)| = 0.707
wbt = w[np.flatnonzero(np.abs(T) < (1/np.sqrt(2)))[0]]

print "wb:", wb
print "wc:", np.round(wc, 2)
print "wbt:", np.round(wbt, 2)
if (PM < 90) and (wb < wc) and (wc < wbt):
    print "Frequency range wb < wc < wbt is valid"


plt.show()
