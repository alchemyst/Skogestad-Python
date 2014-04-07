import numpy as np
import matplotlib.pyplot as plt
from utils import feedback, bodeclosedloop, tf, marginsclosedloop

s = tf([1, 0], 1)
G = 4 /((s - 1) * (0.02 * s + 1)**2)
Kc = 1.25
Tauc = 1.5
K = Kc * (1 + 1 / (Tauc * s))
L = K * G
T = feedback(L, 1)
S = feedback(1, L)


plt.figure('Figure 2.9')
tspan = np.linspace(0, 4, 50)
[t, y] = T.step(0, tspan)
plt.plot(t, y)
[t, y] = S.step(0, tspan)
plt.plot(t, y)
plt.plot([0, 4], np.ones(2))
plt.xlabel('Time [sec]')
plt.legend(['y(t)','u(t)'])
#TODO there is a descrepancy with the u(t) plot

bodeclosedloop(G, K, -1, 2, label='Figure 2.15', margin=True)
#TODO there is a descrepancy with the phase plots

GM, PM, wc, wb, wbt, valid = marginsclosedloop(L) 
print 'GM :' , GM
print "PM:", np.round(PM*180/np.pi, 1), "deg or", np.round(PM, 2), "rad"
print 'wb :' , wb
print 'wc :' , wc
print 'wbt :' , wbt

plt.show()
