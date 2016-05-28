#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def G_P(k, tau, theta, s):
    return k/(tau*s + 1)*np.exp(-theta*s)

def G(s):
    return G_P(2.5, 2.5, 2.5, s)

def w_A(s):
    k = 0.5
    tau1 = 1/0.14
    tau2 = 1/0.25
    tau3 = 1/3.0
    return k*(tau1*s + 1)/(tau2*s + 1)/(tau3*s + 1)

def circle(centerx, centery, radius):
    angle = np.linspace(0, 2*np.pi)
    x = centerx + np.cos(angle)*radius
    y = centery + np.sin(angle)*radius
    plt.plot(x, y, 'r-')

varrange = np.arange(2, 3, 0.1)

N = 1000

for omega in [0.01, 0.05, 0.2, 0.5, 1, 2, 7]:
    s = omega * 1j
    frn = G(s)
    r = abs(w_A(s))

    frp = np.array([G_P(k,tau,theta,s)
                    for k in varrange
                    for tau in varrange
                    for theta in varrange])
    plt.plot(np.real(frp), np.imag(frp), 'b.')

    plt.plot(np.real(frn), np.imag(frn), 'ro')
    circle(np.real(frn), np.imag(frn), r)

omega = np.logspace(-2, 2, 1000)
s = omega*1j
fr = G_P(2.5, 2.5, 2.5, s)
plt.plot(np.real(fr), np.imag(fr), 'r-')
plt.xlabel('Real part (Re)')
plt.ylabel('Imaginary part (Im)')


plt.figure(2)
frn = G(s)
frp = np.array([G_P(k,tau,theta,s)
                for k in varrange
                for tau in varrange
                for theta in varrange])
deviation = np.abs(frp - frn)
distance = np.max(deviation,axis=0)

plt.loglog(omega, distance, 'b--')
plt.loglog(omega, list(map(abs, w_A(s))), 'r')
plt.xlabel('Frequency')
plt.legend(['Distance','w_A'])

plt.show()
