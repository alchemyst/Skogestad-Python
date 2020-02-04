import matplotlib.pyplot as plt
import robustcontrolplot
import numpy as np

from robustcontrol import tf, feedback

s = tf([1, 0])

#Figure 5.8
G = (-s + 1)/(s + 1)
Kcs = [0.1, 0.5, 0.9]
K = -1*s/((1 + 0.02*s)*(1 + 0.05*s))

plt.figure('Figure 5.8')
robustcontrolplot.freq_step_response_plot(G, K, Kcs, 0.2, 'S')

# Plot negative step response
t = np.linspace(0, 0.2, 1000)
u1 = np.ones(500)
u2 = np.zeros(500)
u = np.hstack((u1, u2))

plt.figure('Figure 5.8(b)')
plt.title('(b) Response to step in reference')
for Kc in Kcs:
    K2 = Kc * K
    T = feedback(G * K2, 1)
    tout, y, _ = T.lsim(u, t)    
    plt.plot(tout, y)

plt.plot(tout, u, color='black', linestyle='--', linewidth=0.25)
plt.legend(["Kc = %1.1f" % Kc for Kc in Kcs])
plt.ylabel('y(t)')
plt.xlabel('Time [s]')
plt.show()
