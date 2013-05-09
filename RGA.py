import numpy as np
import matplotlib.pyplot as plt
from utils import RGA

# example 3.9 Skogestad pg 85

A = np.matrix([[1, 1], [0.4, -0.1]])

print RGA(A)

#  the next two function is to calculate the frequency dependent RGA


def G(w):
    """ function to create the matrix of transfer functions"""
    s = w*1j

    #  the matrix transfer function
    #  this specific one is Example 3.11
    G = 1/(5*s + 1)*np.matrix([[s + 1, s + 4],
                                [1, 2]])
    return G


def RGA_alt(G):
    """instead of using util's"""
    return np.array(G)*np.array(G.I).T


def RGA_w(w_start, w_end, x, y):
    """ w_start is the start of logspace
    w_end is the ending of the logspace
    x and y is refer to the indices of the RGA matrix that needs to be plotted

    this is to calculate the RGA at different frequencies
    this give more conclusive values of which pairing would give fast responses
    under dynamic situations"""

    w = np.logspace(w_start, w_end, 1000)
    store = np.zeros([len(x), len(w)])

    count = 0
    for w_i in w:
        RGA_w = RGA(G(w_i))
        store[:, count] = RGA_w[x, y]
        count = count + 1

    for i in range(len(x)):
        plt.loglog(w, store[i, :])

    plt.title('RGA over Freq')
    plt.xlabel('w')
    plt.ylabel('|RGA values| given x , y ')
    plt.show()


def RGA_all(w_start, w_end):
    """w_start is the start of logspace
    w_end is the end of the logspace

    Function that creates a subplot of all the RGA elements as a function of
    frequency"""
    RGAs = []
    w = np.logspace(w_start, w_end, 1000)
    for w_i in w:
        RGAs.append(RGA_alt(G(w_i)))

    RGAs = np.array(RGAs)

    siz = RGAs.shape[2]
    f, ax = plt.subplots(siz, siz)

    for i in range(0, siz):
        for j in range(0, siz):
            ax[i, j].semilogx(w, RGAs[:, i, j])
            if i == siz - 1:
                ax[i, j].set_xlabel('Frequency')

            lab = 'RGA ' + str(i + 1) + ', ' + str(j + 1)
            ax[i, j].set_title(lab)

    f.tight_layout()
    plt.show()


RGA_w(-6, 2, [0, 1], [0, 0])
RGA_all(-2, 2)
