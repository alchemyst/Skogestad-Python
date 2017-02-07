import numpy as np
import matplotlib.pyplot as plt

w = np.logspace(-2, 1, 100)
s = w * 1j

plt.figure('Figure 7.6')

T = 4
wI1 = (T * s + 0.2) / (T/2.5 * s + 1)
plt.loglog(w, np.abs(wI1), "r")

wI2 = wI1 * (s**2 + 1.6 * s + 1) / (s**2 + 1.4 * s + 1)
plt.loglog(w, np.abs(wI2), "--",color='lime',linewidth=2)

k = 2.5
tau = 2.5
Gp0 = Gp = k / (tau * s + 1)

pars = [2, 2.5, 3]
for p1 in range(0, 3):
    theta = pars[p1]
    for p2 in range(0, 3):
        k = pars[p2]
        for p3 in range(0, 3):
            tau = pars[p3]
            Gp = k / (tau * s + 1) * np.exp(-theta * s)
            lI = np.abs((Gp - Gp0) / Gp0)
            plt.loglog(w, lI, "b-.")


plt.ylabel('Magnitude')
plt.xlabel('Frequency')
plt.legend([r'$W_{I1}$', r'$W_{I2}$', 'Perturbed plants'],
           bbox_to_anchor=(0, 1.01, 1, 0), loc=3, ncol=3)

plt.show()
