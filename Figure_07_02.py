#!/usr/bin/env python

from numpy import *
from matplotlib.pyplot import *

def G_P(k, tau, theta, s):
    return k/(tau*s + 1)*exp(-theta*s)

def G(s):
    return G_P(2.5, 2.5, 2.5, s)

def w_A(s):
    k = 0.5
    tau1 = 1/0.14
    tau2 = 1/0.25
    tau3 = 1/3.0
    return k*(tau1*s + 1)/(tau2*s + 1)/(tau3*s + 1)

def circle(centerx, centery, radius):
    angle = linspace(0, 2*pi)
    x = centerx + cos(angle)*radius
    y = centery + sin(angle)*radius
    plot(x, y, 'r-')

def randomparameters():
    return [2 + random.rand() for i in range(3)]

w = 0.2
N = 1000

for w in [0.01, 0.05, 0.2, 0.5, 1, 2, 7]:
    for i in xrange(N):
        s = w*1j
        k, tau, theta = randomparameters()
        frp = G_P(k, tau, theta, s)
        frn = G(s)
        plot(real(frp), imag(frp), 'b.')
        plot(real(frn), imag(frn), 'ro')
        r = abs(w_A(s))
        circle(real(frn), imag(frn), r)

w = logspace(-2, 2, 100)
s = w*1j
fr = G_P(2.5, 2.5, 2.5, s)
plot(real(fr), imag(fr), 'r-')

figure(2)
frn = G(s)
distance = zeros_like(w)
for i in xrange(N):
    k, tau, theta = randomparameters()
    frp = G_P(k, tau, theta, s)
    distance = maximum(abs(frp - frn), distance)

loglog(w, distance, 'b--')
loglog(w, map(abs, w_A(s)), 'r')

show()
