"""
Figure 5.18 Illustration of controllability requirments

Legend
------
M1: Margin to stay within constraints, |u| < 1
M2: Margin for performance, |e| < 1
M3: Margin because of RHP-pole, p
M4: Margin because of RHP-zero, z
M5: Margin because of phase lag, angle-G(jwu) = -180 deg
M6: Margin because of delay, theta
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import tf, margins
import Chapter_05 as ch5

s = tf([1, 0], 1)

# Example plant based on Example 2.9 and Example 2.16
G = (s + 200) / ((10 * s + 1) * (0.05 * s + 1)**2)
G.deadtime = 0.002
Gd = 33 / (10 * s + 1)
K = 0.4 * ((s + 2) / s) * (0.075 * s + 1)

L = G * K

w = np.logspace(-3,3)

GM, PM, wc, wu = margins(G)
GM, PM, wd, w_180 = margins(Gd)

[valid6, wz] = ch5.rule6(G, 1)
[valid5, wtd] = ch5.rule5(G)
[valid8, wp] = ch5.rule8(G)

print 'wc: %0.2f' % wc
print 'wd: %0.2f' % wd
print 'wu: %0.2f' % wu

wuy = abs(G((1j*wu)))
wzy = abs(G((1j*wz)))
wpy = abs(G((1j*wp)))
wtdy = abs(G((1j*wtd)))

m1x = 10**-1
m1g = abs(G(1j*m1x))
m1gd = abs(Gd(1j*m1x))

m2x = 10**2.5
m2gd = abs(Gd(1j*m2x))
m2l = abs(L(1j*m2x))

plt.figure('Figure 5.18')

p1, = plt.loglog(w, abs(G(1j*w)))
p2, = plt.loglog(w, abs(Gd(1j*w)))
p3, = plt.loglog(w, abs(L(1j*w)))

plt.plot(w, 1 * np.ones(len(w)), ls='-.')

a1, = plt.plot([wc, wc], [10**-6, 1], ls='--')
a2, = plt.plot([wd, wd], [10**-6, 1], ls='--')
a3, = plt.plot([wu, wu], [10**-6, wuy], ls='--')
a4, = plt.plot([wp, wp], [10**-6, wpy], ls='--')
a5, = plt.plot([wz, wz], [10**-6, wzy], ls='--')
a6, = plt.plot([wtd, wtd], [10**-6, wtdy], ls='--')

plt.annotate('', (m1x, m1g), (m1x, m1gd), arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))
plt.annotate('$M_1$', (m1x + 0.01, abs(m1g - m1gd) / 3))

plt.annotate('', (m2x, m2gd), (m2x, m2l), arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))
plt.annotate('$M_2$', (m2x + 25, abs(m2gd - m2l) / 1.5))

if (wp != 0):
    plt.annotate('', (wp, 10**-2.5), (wc, 10**-2.5), arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))
    plt.annotate('$M_3$', (abs(wc - wp) / 1.5, 10**-2.3))

if (wz != 0):
    plt.annotate('', (wc, 10**-3.5), (wz, 10**-3.5), arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))
    plt.annotate('$M_4$', (abs(wc - wz) / 3, 10**-3.3))

plt.annotate('', (wc, 10**-4.5), (wu, 10**-4.5), arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))
plt.annotate('$M_5$', (abs(wc - wu) / 0.5, 10**-4.3))

plt.annotate('', (0.001, 10**-2.5), (wd, 10**-2.5), arrowprops=dict(arrowstyle='<-', shrinkA=0, shrinkB=0))
plt.annotate('Control needed to reject disturbances', (0.0015, 10**-2.3))

if (wtd != 0):
    plt.annotate('', (wc, 10**-5.5), (wtd, 10**-5.5), arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))
    plt.annotate('$M_6$', (abs(wc - wtd) / 8, 10**-5.3))

l1 = plt.legend((p1, p2, p3), ['$G$', '$G_d$', '$L$'], bbox_to_anchor=(0, 1.01, 1, 0), loc=3, ncol=3)
l2 = plt.legend((a1, a2, a3, a4, a5, a6), ['$\omega_c$', '$\omega_d$', '$\omega_u$', '$2p$', '$z/2$', '$1/theta$'], loc=1)
plt.gca().add_artist(l1) # add l1 as a separate artist to the axes

plt.xlabel("Frequency [rad/s]")  
plt.ylabel("Magnitude")
plt.show()
