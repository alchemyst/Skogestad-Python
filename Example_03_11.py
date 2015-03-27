import numpy as np
import matplotlib.pyplot as plt
from utils import RGA, RGAnumber
from utilsplot import rga_plot, rga_nm_plot

# The following code performs Example 3.11 of Skogestad.
def G(s):
    G = 0.01**(-5*s)/((s + 1.72e-4)*(4.32*s + 1))*np.matrix([[-34.54*(s + 0.0572), 1.913], [-30.22*s, -9.188*(s + 6.95e-4)]])
    return G

I1 = np.identity(2)
I2 = np.array([[0, 1], [1, 0]])

w = np.logspace(-5, 1, 1000)
s = 1j*w
freq = map(G, s)

rgas = [R for R in map(RGA, freq)]

#plots from textbook
plt.figure('Figure 3.8')
plt.subplot(1, 2, 1)
plt.title('(a) Magnitudeof RGA elements')
plt.semilogx(w, [np.abs(rgas[i][0, 0]) for i in range(0, len(w))], 'r')
plt.semilogx(w, [np.abs(rgas[i][0, 1]) for i in range(0, len(w))], 'b')
plt.text(3e-4, 0.8, '|$\lambda$$_1$$_2$| = |$\lambda$$_2$$_1$|', fontsize=15)
plt.text(3e-4, 0.2, '|$\lambda$$_1$$_1$| = |$\lambda$$_2$$_2$|', fontsize=15)
plt.ylabel('|$\lambda$$_{i,j}$|')
plt.xlabel('Frequency [rad/sec]')

plt.subplot(1, 2, 2)
plt.title('(b) RGA numbers')
plt.semilogx(w, [RGAnumber(Gfr, I1) for Gfr in freq], 'r')
plt.semilogx(w, [RGAnumber(Gfr, I2) for Gfr in freq], 'b')
plt.text(1e-4, 3.2, 'Diagonal pairing', fontsize=15)
plt.text(1e-4, 0.5, 'Off-diagonal pairing', fontsize=15)
plt.ylabel('||$\Lambda$(G) - I||$_{sum}$')
plt.xlabel('Frequency [rad/sec]')
plt.show()

#expanded RGA plots
rga_plot(G, -5, 1)
rga_nm_plot(G, I1, -5, 1)
rga_nm_plot(G, I2, -5, 1)
