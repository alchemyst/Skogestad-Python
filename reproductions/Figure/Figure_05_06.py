import numpy as np
import matplotlib.pyplot as plt


# This module generates the graph in fig 5.6
def S(s, theta):
    S = 1 - np.exp(-theta*s)
    return S


w = np.logspace(-2, 2, 1000)
theta = 0.5

# Generate the Magnitude|S|
plt.loglog(w, np.abs([S((1j*i), theta) for i in w]))

# Generate w = 1/theta
plt.vlines(1/theta, 1e-2, 1, 'k')

# Generate magnitude = 1
plt.hlines(1, 1e-2, 1e2, 'k')

# Set plot properties
plt.title('Figure 5.6')
plt.text(1, 0.7e-1, r'$\omega$=1/$\theta$', fontsize=15)
plt.xlim(1e-2, 1e2)
plt.ylim(1e-2, 1e1)
plt.show()
