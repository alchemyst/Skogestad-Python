import numpy as np
import matplotlib.pyplot as plt
from utils import feedback, tf

s = tf([1, 0], 1)

G = 5/((10*s+1)*(s+1))
Gd = 0.5/((s+1)*(0.2*s+1))
K = 0.04 / s * ((10 * s + 1)**2)/((0.1 * s + 1)**2)

L = G * K

T = feedback(L, 1)
S = feedback(1, L)
    
plt.figure('Figure 5.16 (a)')

plt.subplot(2, 1, 1)
w = np.logspace(-2, 1, 1000)
wi = w*1j 
plt.loglog(w, np.abs(G(wi)))
plt.loglog(w, np.abs(Gd(wi)))
plt.loglog(w, np.ones(len(w)), '-.')
plt.title('(a) $|G|$ & $|G_d|$ with $k_d$=0.5')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude') 
plt.legend(('$|G|$','$|G_d|$'), loc=1)

#TODO plot the correct constraint and unconstraint response
plt.subplot(2, 1, 2)
tspan = np.linspace(0, 10, 100)
[t, y] = T.step(0, tspan)
[t, u] = S.step(0, tspan)
plt.plot(t, y)
plt.plot(t, u)
plt.plot(tspan, 0 * np.ones(100), ls='--')
plt.xlabel('Time [sec]')
plt.legend(['$y(t)$','$u(t)$'])
plt.title('(b) Response to step in disturbance ($k_d=0.5$)')

plt.show()    
