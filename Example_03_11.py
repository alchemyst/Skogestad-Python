import numpy as np
import matplotlib.pyplot as plt
from utils import RGA

# The following code performs Example 3.11 of Skogestad.


def G(s):
    """ Steady state model of fluid catalytic process """
    K = 0.01*np.exp(-5*s)/((s + 1.72e-4)*(4.32*s + 1))
    G = K*np.array([[-34.54*(s + 0.0572), 1.913],
                    [-30.22*s, -9.188*(s + 6.95e-4)]])
    return G


def RGAnumberDiag(A):
    RGAnumD = np.sum(np.abs(RGA(A) - np.identity(len(A))))
    return RGAnumD


def RGAnumberoffDiag(A):
    RGAnumOD = np.sum(np.abs(RGA(A) - np.array([[0, 1], [1, 0]])))
    return RGAnumOD

w = np.logspace(-5, 1, 1000)
s = 1j*w
freq = map(G, s)

l = [R for R in map(RGA, freq)]
Diagnum = np.array([Rd for Rd in map(RGAnumberDiag, freq)])
offDiagnum = np.array([Rod for Rod in map(RGAnumberoffDiag, freq)])

ax = plt.subplot(1, 2, 1)
plt.semilogx(w, [np.abs(l[i][0, 0]) for i in range(0, len(w))], 'r',
             label='|$\lambda$$_1$$_2$| = |$\lambda$$_2$$_1$|')
plt.semilogx(w, [np.abs(l[i][0, 1]) for i in range(0, len(w))], 'b',
             label='|$\lambda$$_1$$_1$| = |$\lambda$$_2$$_2$|')
plt.title('(a)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.15,
                 box.width, box.height * 0.9])

ax = plt.subplot(1, 2, 2)
plt.semilogx(w, [Diagnum[i] for i in range(0, len(w))], 'r',
             label='Diagonal pairing')
plt.semilogx(w, [offDiagnum[i] for i in range(0, len(w))], 'b',
             label='Off-diagonal pairing')
plt.title('(b)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.15,
                 box.width, box.height * 0.9])
plt.show()
