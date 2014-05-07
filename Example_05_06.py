
import matplotlib.pyplot as plt
import numpy as np
from utils import tf, feedback

# Process model transfer function
s = tf([1, 0])
G = (-s + 1)/(s + 1)

# Controllers transfer function with various controller gains
Ks = [Kc * ((s + 1)/s) * (1 / (0.05 * s + 1)) for Kc in [0.2, 0.5, 0.8]]

# Complementary sensitivity transfer functions
Ts = [feedback(G * K, 1) for K in Ks]

plt.figure('Figure 5.7')
w = np.logspace(-2, 2, 1000)
wi = w * 1j
plt.subplot(1, 2, 1)
plt.title('(a) Sensitivity function')
for T in Ts:
    S = 1 - T
    plt.loglog(w, abs(S(wi)))
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude |S|')
plt.legend(["Kc=0.2", "Kc=0.5", "Kc=0.8"],
           bbox_to_anchor=(0, 1.03, 1, 0), loc=3, ncol=3)
           
plt.subplot(1, 2, 2)
plt.title('(b) Response to step in reference')
tspan = np.linspace(0, 5, 100)
for T in Ts:
    [t, y] = T.step(0, tspan)
    plt.plot(t, y)
plt.xlabel('Time [sec]')
plt.ylabel('y(t)')

plt.show()
