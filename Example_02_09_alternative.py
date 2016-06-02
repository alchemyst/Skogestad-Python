from __future__ import print_function
from utils import tf, feedback, tf_step
import utilsplot
import matplotlib.pyplot as plt
import numpy as np

s = tf([1,0], 1)
G = 200/((10*s + 1)*(0.05*s + 1)**2)
Gd = 100/(10*s + 1)

wc = 10
K = wc*(10*s + 1)*(0.1*s + 1)/(200*s*(0.01*s + 1))
L = G*K

t = np.linspace(0, 3)
Sd = (1/(1 + G*K))*Gd
T = feedback(L, 1)

[t,y] = tf_step(T, 3)
plt.figure('Figure 2.22')
plt.subplot(1, 2, 1)
plt.plot(t, y)
plt.title('Tracking Response')
plt.ylabel('y(t)')
plt.xlabel('Time (s)')
plt.ylim([0, 1.5])

[t,yd] = Sd.step(0, t)
plt.subplot(1, 2, 2)
plt.plot(t, yd)
plt.ylabel('y(t)')
plt.xlabel('Time (s)')
plt.title('Disturbance Response')
plt.ylim([0, 1.5])
plt.show()
