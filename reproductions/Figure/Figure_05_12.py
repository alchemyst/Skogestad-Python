import robustcontrolplot
import matplotlib.pyplot as plt

from robustcontrol import tf

s = tf([1, 0])

# Figure 5.12
G = 4/((s - 1)*(0.02*s + 1)**2)
Kc = [0.5, 1.25, 2.0]
tau = 1.25
K = (tau*s + 1)/(tau*s)

plt.figure('Figure 5.12')
robustcontrolplot.freq_step_response_plot(G, K, Kc, 4, 'T')
plt.show()
