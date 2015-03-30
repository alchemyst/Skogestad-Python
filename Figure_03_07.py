import numpy as np
import matplotlib.pyplot as plt

from utilsplot import sv_plot, condtn_nm_plot

def G1(s):
    """def the function for Figure 3.7)a the distilation process"""
    return 1/(75*s + 1)*np.matrix([[87.8, -86.4], 
                                  [108.2, -109.6]])

def G2(s):
    """def the function for Figure 3.7)b the spinning satelite"""
    return 1/(s**2 + 10**2)*np.matrix([[s - 1e2, 10*(s + 1)], 
                                       [-10*(s + 1), s - 1e2]])

processes = [[G1, 'Distillation process 3.7(a)', -4, 1],
             [G2, 'Spinning satellite 3.7(b)', -2, 2]]

plt.figure('Figure 3.7')
for i, [G, title, minw, maxw] in enumerate(processes):
    # Singular values
    omega = np.logspace(minw, maxw, 1000)
    s = 1j * omega
    Gw = map(G, s)
    plt.subplot(2, 2, i + 1)
    plt.title(title)
    sv_plot(G, minw, maxw)
    
    """this is an additional plot to the textbook"""
    plt.subplot(2, 2, 3 + i)
    condtn_nm_plot(G, minw, maxw)

plt.show()
