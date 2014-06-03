import numpy as np
import matplotlib.pyplot as plt
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


plt.rc('text', usetex=True)

processes = [[G1, 'Distillation process 3.7(a)', -4, 1],
             [G2, 'Spinning satellite 3.7(b)', -2, 2]]

# Plotting of Figure 3.7)a and 3.7)b
for i, [G, title, minw, maxw] in enumerate(processes):
    # Singular values
    omega = np.logspace(minw, maxw, 1000)
    s = 1j * omega
    Gw = map(G, s)
    plt.subplot(2, 2, i + 1)
    plt.loglog(omega, map(utils.sigmas, Gw))
    plt.ylabel(r'Singular value magnitude')
    plt.title(title)
    plt.legend([r'$\bar \sigma$(G)',
                   r'$\underline\sigma$(G)'], 'best')
    plt.subplot(2, 2, 3 + i)
    plt.semilogx(omega, map(condition_number, Gw))
    plt.ylabel('Condition number')
    plt.xlabel(r'Frequency [rad/s]')

plt.show()
