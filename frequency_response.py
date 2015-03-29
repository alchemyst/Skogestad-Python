import matplotlib.pyplot as plt

from utilsplot import bode
from utils import tf


s = tf([1, 0])

for power in range(1, 4):
    G = 1/(s + 1)**power
    bode(G)

plt.show()
