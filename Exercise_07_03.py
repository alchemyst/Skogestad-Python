# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:10:25 2015

@author: cronjej
"""

import numpy as np
import matplotlib.pyplot as plt


def G(s):
    return 1/(s + 1)


def wI(s):
    return (s + 0.3)/((1/3)*s + 1)

    
def lI(Gp, G):
    return np.abs((Gp-G)/G)


def satisfy(wI, G, Gp, params, s):
          
    distance = np.zeros((len(params), len(s)))
    distance_min = np.zeros(len(params))
    for i in range(len(params)):
        for j in range(len(s)):
            distance[i, j] = np.abs(wI(s[j])) - lI(Gp(params[i], s[j]), G(s[j]))
        distance_min[i] = np.min(distance[i, :])
    param_range = params[distance_min > 0]
    
    return param_range

    
w = np.logspace(-3, 3, 1000)
s = 1j*w

# a) Ga = G*exp(-theta*s)

def Gp_a(theta, s):
    return G(s)*np.exp(-theta*s)

thetas = np.linspace(0, 2, 201)
theta_range = satisfy(wI, G, Gp_a, thetas, s)
theta_max = np.max(theta_range)
print "a) %s" % theta_max

plt.figure(1)
plt.loglog(w, np.abs(wI(s)), label='$w_{I}$')
plt.loglog(w, lI(Gp_a(theta_max, s), G(s)), label='$l_{I}$')
plt.xlabel('Frequency  [rad/s]')
plt.ylabel('Magnitude')
plt.legend(loc=2)

# b) Gb = G*(1/(tau*s+1))

def Gp_b(tau, s):
    return G(s)/(tau*s + 1)
 
taus = np.linspace(0, 2, 201)
taus_range = satisfy(wI, G, Gp_b, taus, s)
tau_max = np.max(taus_range)
print "b) %s" % tau_max

plt.figure(2)
plt.loglog(w, np.abs(wI(s)), label='$w_{I}$')
plt.loglog(w, lI(Gp_b(tau_max, s), G(s)), label='$l_{I}$')
plt.xlabel('Frequency  [rad/s]')
plt.ylabel('Magnitude')
plt.legend(loc=2)

# c) Gc = 1/(s-a)

def Gp_c(a, s):
    return 1/(s + a)

poles = np.linspace(0, 2, 201)
pole_range = satisfy(wI, G, Gp_c, poles, s)
print "c) %s to" % str(np.min(pole_range)), str(np.max(pole_range))

plt.figure(3)
plt.loglog(w, np.abs(wI(s)), label='$w_{I}$')
plt.loglog(w, lI(Gp_c(np.min(pole_range), s), G(s)), label='$l_{I}(p_{min})$')    
plt.loglog(w, lI(Gp_c(np.max(pole_range), s), G(s)), label='$l_{I}(p_{max})$')   
plt.xlabel('Frequency  [rad/s]')
plt.ylabel('Magnitude')
plt.legend(loc=2)

# d) Gd = 1/(Ts+1)

def Gp_d(T, s):
    return 1/(T*s + 1)

Taus = np.linspace(0, 3, 301)
Tau_range = satisfy(wI, G, Gp_d, Taus, s)
print "d) %s to" % str(np.min(Tau_range)), str(np.max(Tau_range))

plt.figure(4)
plt.loglog(w, np.abs(wI(s)), label='$w_{I}$')
plt.loglog(w, lI(Gp_d(np.min(Tau_range), s), G(s)), label=r'$l_{I}(\tau_{min})$')    
plt.loglog(w, lI(Gp_d(np.max(Tau_range), s), G(s)), label=r'$l_{I}(\tau_{max})$')
plt.xlabel('Frequency  [rad/s]')
plt.ylabel('Magnitude')
plt.legend(loc=2)

# e) Ge = G*(1 / ((s/70)**2 + 2*zeta*(s/10) + 1))  *Middle term in book incorrect

def Gp_e(zeta, s):
    return G(s)/((s/70)**2 + 2*zeta*(s/10) + 1)

zetas = np.linspace(0, 8, 801)
zeta_range = satisfy(wI, G, Gp_e, zetas, s)
print "e) %s to" % str(np.min(zeta_range)), str(np.max(zeta_range))

plt.figure(5)
plt.loglog(w, np.abs(wI(s)), label='$w_{I}$')
plt.loglog(w, lI(Gp_e(np.min(zeta_range), s), G(s)), label='$l_{I}(\zeta_{min})$')    
plt.loglog(w, lI(Gp_e(np.max(zeta_range), s), G(s)), label='$l_{I}(\zeta_{max})$')
plt.xlabel('Frequency  [rad/s]')
plt.ylabel('Magnitude')
plt.legend(loc=2)

# f) Gf = G*(1/(0.01*s+1))**m

def Gp_f(m, s):
    return G(s)*(1/(0.01*s + 1))**m
 
ms = np.linspace(100, 150, 51)
m_range = satisfy(wI, G, Gp_f, ms, s)
m_max = np.max(m_range)
print "f) %s" % m_max

plt.figure(6)
plt.loglog(w, np.abs(wI(s)), label='$w_{I}$')
plt.loglog(w, lI(Gp_f(m_max, s), G(s)), label='$l_{I}$')
plt.xlabel('Frequency  [rad/s]')
plt.ylabel('Magnitude')
plt.legend(loc=2)

# g) Gg = G*(-tauz*s + 1)/(tauz*s + 1)

def Gp_g(tauz, s):
    return G(s)*(-tauz*s + 1)/(tauz*s + 1)

tauzs = np.linspace(0, 1, 101)
tauz_range = satisfy(wI, G, Gp_g, tauzs, s)
tauz_max = np.max(tauz_range)
print "f) %s" % tauz_max

plt.figure(7)
plt.loglog(w, np.abs(wI(s)), label='$w_{I}$')
plt.loglog(w, lI(Gp_g(tauz_max, s), G(s)), label='$l_{I}$')
plt.xlabel('Frequency  [rad/s]')
plt.ylabel('Magnitude')
plt.legend(loc=2)
