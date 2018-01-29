import matplotlib.pyplot as plt

from utils import tf
from utilsplot import bode

# Trasfer function of L(s)
s = tf([1, 0], 1)
L = 3*(-2*s + 1)/((10*s + 1)*(5*s + 1))

plt.figure('Figure 2.7')
plt.title('Bode plots of L(s) with Kc = 1')
bode(L, -2, 1)
plt.show()
