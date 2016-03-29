import numpy as np
import matplotlib.pyplot as plt

from utilsplot import rga_plot, rga_nm_plot


def G(s):
    G = 0.01 ** (-5 * s) / ((s + 1.72e-4) * (4.32 * s + 1)) * np.matrix(
        [[-34.54 * (s + 0.0572), 1.913], [-30.22 * s, -9.188 * (s + 6.95e-4)]])
    return G


I1 = np.asmatrix(np.identity(2))
I2 = np.matrix([[0, 1], [1, 0]])

plt.figure('Figure 3.8')
plt.subplot(1, 2, 1)
plt.title('(a) Magnitude of RGA elements')
rga_plot(G, -5, 1, [None, None, 0, 1], plot_type='all')
plt.text(3e-4, 0.8, '|$\lambda$$_1$$_2$| = |$\lambda$$_2$$_1$|', fontsize=15)
plt.text(3e-4, 0.2, '|$\lambda$$_1$$_1$| = |$\lambda$$_2$$_2$|', fontsize=15)

plt.subplot(1, 2, 2)
plt.title('(b) RGA numbers')
rga_nm_plot(G, [I1, I2], ['Diagonal pairing', 'Off-diagonal pairing'], -5, 1, plot_type='all')
plt.show()

# The section below demonstrates more utilsplot functions
plt.figure('RGA per element')
rga_plot(G, -5, 1, plot_type='elements')

plt.figure('RGA per output')
rga_plot(G, -5, 1, plot_type='outputs')

plt.figure('RGA per input')
rga_plot(G, -5, 1, plot_type='inputs')
plt.show()
