import matplotlib.pyplot as plt
import numpy as np

from utils import feedback, tf, marginsclosedloop, ControllerTuning
from utilsplot import step_response_plot, bodeclosedloop


s = tf([1, 0], 1)
G = 3 * (-2 * s + 1) / ((10 * s + 1) * (5 * s + 1))

[Kc, Taui, Ku, Pu] = ControllerTuning(G, method='ZN') 
print 'Kc: %0.2f' % (Ku/2.2)
print 'Taui: %0.1f' % (Pu/1.2)
print 'Ku: %0.1f' % Ku
print 'Pu: %0.1f' % Pu

K1 = Kc * (1 + 1 / (Taui * s))
K = K1[0] # use this code to remove array
L = G * K
T = feedback(L, 1)
S = feedback(1, L)
u = S * K

plt.figure('Figure 2.8')
step_response_plot(T, u, 80, 0)
plt.show()

plt.figure('Figure 2.14')
bodeclosedloop(G, K, -2, 1, margin=True)
plt.show()

GM, PM, wc, wb, wbt, valid = marginsclosedloop(L) 
print 'GM: %0.2f' % GM
print "PM: %0.2f deg or %0.2f rad" % (PM, PM / 180 * np.pi)
print 'wb: %0.2f' % wb
print 'wc: %0.2f' % wc
print 'wbt: %0.2f' % wbt

if valid: print "Frequency range wb < wc < wbt is valid"
else: print "Frequency range wb < wc < wbt is not valid"
