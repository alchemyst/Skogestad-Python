from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from utils import feedback, tf, marginsclosedloop
from utilsplot import step_response_plot, bodeclosedloop


s = tf([1, 0], 1)
G = 4/((s - 1)*(0.02*s + 1)**2)
Kc = 1.25
Tauc = 1.5
K = Kc*(1 + 1/(Tauc*s))
L = K * G
T = feedback(L, 1)
S = feedback(1, L)
u = S * K

plt.figure('Figure 2.9')
step_response_plot(T, u, 4, 0)
plt.show()

plt.figure('Figure 2.15')
bodeclosedloop(G, K, -1, 2, margin=True)
plt.show()
# TODO there is a discrepancy with the phase plots

GM, PM, wc, wb, wbt, valid = marginsclosedloop(L)
print('GM:', np.round(GM, 2))
print('PM:', np.round(PM / 180 * np.pi, 2), "rad or", np.round(PM, 2), "deg")
print('wb:', np.round(wb, 2))
print('wc:', np.round(wc, 2))
print('wbt:', np.round(wbt, 4))
