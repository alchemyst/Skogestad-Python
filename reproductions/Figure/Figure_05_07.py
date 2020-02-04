import matplotlib.pyplot as plt
import robustcontrolplot

from robustcontrol import tf

s = tf([1, 0])

#Figure 5.7
G = (-s + 1)/(s + 1)
Kc = [0.2, 0.5, 0.8]
K = ((s + 1)/s) * (1/(0.05*s + 1))

plt.figure('Figure 5.7')
robustcontrolplot.freq_step_response_plot(G, K, Kc, 5, 'S')
plt.show()
