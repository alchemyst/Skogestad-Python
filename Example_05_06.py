import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scs
from utils import tf, feedback

# Process model of G with various Controller Gains
s = tf([1, 0])
G = (-s+1)/(s+1)

# Controllers
Ks = [Kc*((s+1)/s)*(1/(0.05*s+1)) for Kc in [0.2, 0.5, 0.8]]

# Closed loop transfer functions
Ts = [feedback(G*K, 1) for K in Ks]

#  The time domian response
plt.subplot(2, 1, 1)
tspan = np.linspace(0, 5, 100)
for T in Ts:
    [t, y] = T.step(0, tspan)
    plt.plot(t, y)
plt.xlabel('Time (sec)')
plt.ylabel('y(t)')

# sensitivity function
w = np.logspace(-2, 2, 1000)
wi = w*1j
plt.subplot(2, 1, 2)
for T in Ts:
    S = 1 - T
    plt.loglog(w, abs(S(wi)))

plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude (S)')
plt.show()
