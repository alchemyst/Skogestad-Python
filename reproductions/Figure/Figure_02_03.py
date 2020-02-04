import matplotlib.pyplot as plt

from robustcontrol import tf
from robustcontrolplot import bode

s = tf([1,0], 1)
G = 30*(s + 1)/((s + 10)*(s + 0.01)**2)

plt.figure('Figure 2.3')
bode(G, -3, 3)
plt.show()
