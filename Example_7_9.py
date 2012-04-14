import numpy as np
import matplotlib.pyplot as plt

def G(s):
    G = 3*(-2*s + 1)/((5*s + 1)*(10*s + 1))
    return G

def K(s, kc):
    K = kc*(12.7*s + 1)/(12.7*s)
    return K

def Wi(s):
    Wi = (10*s + 0.33)/((10/5.25)*s + 1)
    return Wi

def Gnom(s):
    Gnom = 4*(-3*s + 1)/((4*s + 1)**2)
    return Gnom

def l(Gn, G):
    l = np.abs((Gn - G)/G)
    return l

def T(G, K):
    T = (G*K)/(1 + G*K)
    return T

w = np.logspace(-3, 1, 300)
s = 1j*w

plt.figure(0)
plt.loglog(w, l(Gnom(s), G(s)), 'r', label='Relative Error')
plt.loglog(w, np.abs(Wi(s)), 'k', label='$W_I$')
plt.title(r'Figure 7.12')
plt.xlabel(r'Frequency [rad/s]', fontsize=14)
plt.ylabel(r'Magnitude', fontsize=15)
plt.legend()

plt.figure(1)
plt.loglog(w, np.abs(T(G(s), K(s, 1.13))), label='$T_1$ (not RS)')
plt.loglog(w, np.abs(T(G(s), K(s, 0.31))), label='$T_2$')
line = plt.loglog(w, 1/np.abs(Wi(s)), label='$1/W_I$')
plt.title(r'Figure 7.13')
plt.xlabel(r'Frequency [rad/s]', fontsize=14)
plt.ylabel(r'Magnitude', fontsize=15)
plt.legend()
plt.setp(line, linestyle='--')

plt.show()
