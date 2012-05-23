import numpy as np
import matplotlib.pyplot as plt
from utils import tf

s = tf([1, 0], 1)
"""Set-up plant and Disturbance model"""
G = 200/((10*s + 1)*(0.05*s + 1)*(0.05*s + 1))
G_d = 100/(10*s + 1)

"""The four different controllers in question"""
K0 = ((10*(10*s + 1))*(0.1*s + 1))/(s*200*(0.01*s + 1))
K1 = 0.5
K2 = 0.5*(s + 2)/s
K3 = (0.5*(s + 2)*(0.05*s + 1))/(s*(0.005*s + 1))

w = np.logspace(-2, 2, 100)
s = 1j*w

"""Closed loop tf"""
L0 = G*K0
L1 = G*K1
L2 = G*K2
L3 = G*K3

tspan = np.linspace(0, 3, 100)

"""Step response in the disturbance"""
cont0 = (1/(1+L0))*G_d
cont1 = (1/(1+L1))*G_d
cont2 = (1/(1+L2))*G_d
cont3 = (1/(1+L3))*G_d

[t, y0] = cont0.step(0, tspan)
[t, y1] = cont1.step(0, tspan)
[t, y2] = cont2.step(0, tspan)
[t, y3] = cont3.step(0, tspan)

"""Plot results"""
plt.subplot(1, 2, 1)
plt.loglog(w, np.abs(L0(s)))
plt.loglog(w, np.abs(L1(s)))
plt.loglog(w, np.abs(L2(s)))
plt.loglog(w, np.abs(L3(s)))

plt.subplot(1, 2, 2)
plt.plot(t, y0)
plt.plot(t, y1)
plt.plot(t, y2)
plt.plot(t, y3)
plt.show()
