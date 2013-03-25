import numpy as np
import matplotlib.pyplot as plt

# TODO: This should be reworked!

def Stability(Poles_poly):
    """determine if the characteristic equation of a transfer function is stable
    Poles_poly is a polynomial expansion of the characteristic equation """

    return any(np.real(r) > 0 for r in np.roots(Poles_poly))

def System(KC):
    """giving the characteristic equation in terms of the Kc value"""

    Poles_poly = [50, 15-6*KC, 1+3*KC]
    return Poles_poly

Kc = np.linspace(-2, 3, 200)

# creating a plot to indicate over what range of Kc the system would be stable
vec = [Stability(System(K)) for K in Kc]

plt.plot(Kc, vec, 'rD')
plt.show()
