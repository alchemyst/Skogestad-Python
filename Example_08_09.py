import numpy as np
import utils
import matplotlib.pyplot as plt

tau = 0.5
I = np.identity(2)

def G(s):
    """process transfer matrix"""
    return 1 / (tau * s + 1) * np.matrix([[-87.8, 1.4], 
                                          [-108.2, -1.4]])


def K(s):
    """controller"""
    return (tau * s + 1) / s * np.matrix([[-0.0015, 0], 
                                          [0, -0.075]])


def w_I(s):
    """magnitude multiplicative uncertainty in  each input"""
    return (s + 0.2) / (0.5 * s + 1)


def W_I(s):
    """magnitude multiplicative uncertainty in each input"""
    return np.matrix([[(s + 0.2) / (0.5 * s + 1), 0], 
                      [0, (s + 0.2) / (0.5 * s + 1)]])


def T_I(s):
    return K(s) * G(s) * (I + K(s) * G(s)).I

def M(s):
    return w_I(s) * T_I(s)

frequency = np.logspace(-3, 2, 1000)
s = 1j * frequency

max_singular_value_of_T_I = [max(utils.sigmas(T_I(si))) for si in s]
mu_delta_T_I = [max(np.abs(np.linalg.eigvals(T_I(si)))) for si in s]

print 'Robust Stability (RS) is attained if mu(T_I(jw)) < 1/|w_I(jw)| for all applicable frequency range'

plt.loglog(frequency, max_singular_value_of_T_I, 'b')
plt.loglog(frequency, 1 / np.abs(w_I(s)), 'r')
plt.loglog(frequency, mu_delta_T_I, 'y')
plt.legend(('max_singular_value_of(T_I)', '1/|w_I(jw)|', 'mu(T_I)'), 'best', shadow=False)
plt.xlabel('frequency')
plt.ylabel('Magnitude')
plt.show()
