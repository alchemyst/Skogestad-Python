import matplotlib.pyplot as plt
import numpy as np
from utils import feedback, tf, bodeclosedloop, marginsclosedloop, ZeiglerNichols

s = tf([1, 0], 1)
G = 3 * (-2 * s + 1) / ((10 * s + 1) * (5 * s + 1))


[Kc, Taui, Ku, Pu] = ZeiglerNichols(G) 
print 'Kc:', np.round(Ku, 3)
print 'Taui:', np.round(Pu, 3)
print 'Ku:', np.round(Ku, 3)
print 'Pu:', np.round(Pu, 3)

K1 = Kc * (1 + 1 / (Taui * s))
K = K1[0] # use this code to remove array
L = G * K
T = feedback(L, 1)
S = feedback(1, L)


plt.figure('Figure 2.8')
tspan = np.linspace(0, 80, 100)
[t, y] = T.step(0, tspan)
plt.plot(t, y)
[t, y] = S.step(0, tspan)
plt.plot(t, y)
plt.plot([0, 80], np.ones(2))
plt.xlabel('Time [sec]')
plt.legend(['y(t)','u(t)'])
#TODO there is a descrepancy with the u(t) plot

bodeclosedloop(G, K, -2, 1, 'Figure 2.14', margin=True)


GM, PM, wc, wb, wbt, valid = marginsclosedloop(L) 
print 'GM:' , np.round(GM, 2)
print "PM: ", np.round(PM, 1) , "deg or",  np.round(PM / 180 * np.pi, 2), "rad"
print 'wb:' , np.round(wb, 2)
print 'wc:' , np.round(wc, 2)
print 'wbt:' , np.round(wbt, 2)
#TODO there is a descrepancy with the wbt value

if valid: print "Frequency range wb < wc < wbt is valid"
else: print "Frequency range wb < wc < wbt is not valid"

plt.show()
