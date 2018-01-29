import numpy as np
import matplotlib.pyplot as plt

from utils import tf, feedback

s = tf([1, 0])
G = 4/((s - 1)*(0.02*s + 1)**2)
Kc = 1.25
tau = 1.25
K = Kc * (tau*s + 1)/(tau * s)
T = feedback(G * K, 1)
tspan = np.linspace(0, 1.5, 100)

plt.figure('Figure 5.10')
plt.title('Rise time tr according to RHP-pole defintion')
[t, y] = T.step(0, tspan)
plt.plot(t, y)

# exclude leading zero to elimiate div by zero
# calc rise time
tr = np.min(t[1:-1]/y[1:-1])

plt.plot(t*tr, t, color='black', linestyle='--', linewidth=0.75)
plt.plot([tr, tr], [0, 1], color='black', linestyle='--', linewidth=0.75)
plt.axhline(1, color='black', linestyle='--', linewidth=0.75)
plt.text(tr + 0.005, 0, '$t_{r}$', fontsize=10)
plt.text(0, 1.2, 'Slope = $1/t_{r}$', fontsize=10)
plt.text(1.4, 1.01, '$r$', fontsize=10)
plt.xlabel('Time, t')
plt.ylabel('y(t)')
plt.show()

print('tr = %1.2f' % tr)
