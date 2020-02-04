import matplotlib.pyplot as plt
import numpy as np

from robustcontrol import feedback, tf

# Process model of G with various Controller Gains
s = tf([1, 0], 1)
G = 3*(-2*s + 1)/((10*s + 1)*(5*s + 1))

# Configure plot and generate timespan
tspan = np.linspace(0, 50, 100)
plt.figure('Figure 2.6')
plt.title('Effect of proportional gain Kc on closed loop response')

# Calculate the time domain response
Ks = [0.5, 1.5, 2, 2.5, 3.0]
for K in Ks:
    T = feedback(G * K, 1)
    [t, y] = T.step(0, tspan)
    if K >= 3.0:
        plt.plot(t, y, '-.')
    else:
        plt.plot(t, y)

# Plot the time domain response
plt.legend(["Kc = %1.1f" % K for K in Ks])
plt.xlabel('Time [s]')
plt.ylim(-0.5, 2.5)
plt.show()
