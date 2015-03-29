import matplotlib.pyplot as plt
from utils import tf
from utilsplot import bode

G = tf([5], [10, 1], deadtime=2)

plt.figure('Figure 2.2')
bode(G, -3, 1)
plt.show()
