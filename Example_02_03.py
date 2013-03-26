import numpy as np
import matplotlib.pyplot as plt

# TODO: This should be reworked!
# This version is not really a rework, but provides clearer output


def Stability(Poles_poly):
    """
    Determine if the characteristic equation of a transfer function is stable
    Poles_poly is a polynomial expansion of the characteristic equation
    """
    return any(np.real(r) > 0 for r in np.roots(Poles_poly))


def System(KC):
    # Gives the characteristic equation in terms of the Kc value
    Poles_poly = [50, 15 - 6 * KC, 1 + 3 * KC]
    return Poles_poly

Kc = np.linspace(-2, 3, 1000)

# Creates a plot to indicate over what range of Kc the system would be stable

vec = [Stability(System(K)) for K in Kc]

trigger = 0
for i in range(0, len(Kc) - 1):
    if vec[i] == 1:
        if vec[i] != vec[i - 1]:
            print 'Change from stable to unstable at Kc = ' + '%.2f' % Kc[i]
            Limit1 = Kc[i]
            trigger = 1
    elif vec[i] == 0:
        if vec[i] != vec[i - 1]:
            print 'Change from unstable to stable at Kc = ' + '%.2f' % Kc[i]
            Limit2 = Kc[i]
            trigger = 1

if trigger == 0:
    print 'No stability margins could be found'
else:
    print 'Stable between Kc = ' + '%.2f' % Limit1 \
          + ' and Kc = ' + '%.2f' % Limit2

plt.plot(Kc, vec, 'rD')
plt.show()