import numpy as np
import matplotlib.pyplot as plt
from utils import RGA

# The following code performs Example 3.11 of Skogestad.
def G(s):
    G = 0.01*np.exp(-5*s)/((s + 1.72e-4)*(4.32*s + 1))*np.array([[-34.54*(s + 0.0572), 1.913], [-30.22*s, -9.188*(s + 6.95e-4)]])
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

plt.subplot(1, 2, 1)
plt.semilogx(w, [np.abs(l[i][0, 0]) for i in range(0, len(w))], 'r')
plt.semilogx(w, [np.abs(l[i][0, 1]) for i in range(0, len(w))], 'b')
plt.title('(a)')
plt.text(3e-4, 0.8, '|$\lambda$$_1$$_2$| = |$\lambda$$_2$$_1$|', fontsize=15)
plt.text(3e-4, 0.2, '|$\lambda$$_1$$_1$| = |$\lambda$$_2$$_2$|', fontsize=15)

plt.subplot(1, 2, 2)
plt.semilogx(w, [Diagnum[i] for i in range(0, len(w))], 'r')
plt.semilogx(w, [offDiagnum[i] for i in range(0, len(w))], 'b')
plt.title('(b)')
plt.text(1e-4, 3.2, 'Diagonal pairing', fontsize=15)
plt.text(1e-4, 0.5, 'Off-diagonal pairing', fontsize=15)
plt.show()
