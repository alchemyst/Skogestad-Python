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

def randomparameters():
    return [2 + np.random.rand() for i in range(3)]

N = 1000

for omega in [0.01, 0.05, 0.2, 0.5, 1, 2, 7]:
    s = omega * 1j
    frn = G(s)
    r = abs(w_A(s))

    for i in range(N):
        k, tau, theta = randomparameters()
        frp = G_P(k, tau, theta, s)
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
distance = np.zeros_like(omega)
for i in range(N):
    k, tau, theta = randomparameters()
    frp = G_P(k, tau, theta, s)
    distance = np.maximum(abs(frp - frn), distance)

plt.loglog(omega, distance, 'b--')
plt.loglog(omega, list(map(abs, w_A(s))), 'r')
plt.xlabel('Frequency')
plt.legend(['Distance','w_A'])

plt.show()
