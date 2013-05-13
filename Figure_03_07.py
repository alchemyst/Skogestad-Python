import numpy as np
import numpy.linalg as la
from matplotlib import pyplot
import utils


def G1(s):
    """def the function for Figure 3.7)a the distilation process"""
    return 1/(75*s + 1)*np.matrix([[87.8, -86.4], 
                                  [108.2, -109.6]])

def G2(s):
    """def the function for Figure 3.7)b the spinning satelite"""
    return 1/(s**2 + 10**2)*np.matrix([[s - 1e2, 10*(s + 1)], 
                                       [-10*(s + 1), s - 1e2]])


def condition_number(G):
    """Function to determine condition number"""
    sig = utils.sigmas(G)
    return max(sig) / min(sig)


pyplot.rc('text', usetex=True)

processes = [[G1, 'Distillation process 3.7(a)', -4, 1],
             [G2, 'Spinning satellite 3.7(b)', -2, 2]]

# Plotting of Figure 3.7)a and 3.7)b
for i, [G, title, minw, maxw] in enumerate(processes):
    # Singular values
    omega = np.logspace(minw, maxw, 1000)
    s = 1j * omega
    Gw = map(G, s)
    pyplot.subplot(2, 2, i + 1)
    pyplot.loglog(omega, map(utils.sigmas, Gw))
    pyplot.ylabel(r'Singular value magnitude')
    pyplot.title(title)
    pyplot.legend([r'$\bar \sigma$(G)',
                   r'$\underline\sigma$(G)'], 'best')
    pyplot.subplot(2, 2, 3 + i)
    pyplot.semilogx(omega, map(condition_number, Gw))
    pyplot.ylabel('Condition number')
    pyplot.xlabel(r'Frequency [rad/s]')

pyplot.show()