import numpy as np
import matplotlib.pyplot as plt
from utils import feedback, bodeclosedloop, tf, marginsclosedloop
from utilsplot import step_response_plot

s = tf([1, 0], 1)
G = 4 /((s - 1) * (0.02 * s + 1)**2)
Kc = 1.25
Tauc = 1.5
K = Kc * (1 + 1 / (Tauc * s))
L = K * G
T = feedback(L, 1)
S = feedback(1, L)
u = S * K

plt.figure('Figure 2.9')
step_response_plot(T, u, 4, 0)
plt.show()

bodeclosedloop(G, K, -1, 2, label='Figure 2.15', margin=True)
#TODO there is a descrepancy with the phase plots

GM, PM, wc, wb, wbt, valid = marginsclosedloop(L) 
print 'GM :' , GM
print "PM:", np.round(PM*180/np.pi, 1), "deg or", np.round(PM, 2), "rad"
print 'wb :' , wb
print 'wc :' , wc
print 'wbt :' , wbt
