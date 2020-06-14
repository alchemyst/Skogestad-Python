import numpy as np
import matplotlib.pyplot as plt

from utilsplot import plot_freq_subplot, sv_dir_plot
from utils import mimotf, tf


def SVD_w(G, w_start=-2, w_end=2, axlim=None, points=10000):
    """ 
    Singular value demoposition functions    
    """

    #  this is an open loop study

    #  w_end = end of the logspace for the frequency range

    w = np.logspace(w_start, w_end, points)

    #  getting the size of the system
    A = G(0.0001)
    [U, s, V] = np.linalg.svd(A)
    output_direction_max = np.zeros([len(w), U.shape[0]])
    input_direction_max = np.zeros([len(w), V.shape[0]])
    output_direction_min = np.zeros([len(w), U.shape[0]])
    input_direction_min = np.zeros([len(w), V.shape[0]])

    store_max = np.zeros(len(w))
    store_min = np.zeros(len(w))
    count = 0
    for w_iter in w:
        A = G(w_iter)
        [U, S, V] = np.linalg.svd(A)

        output_direction_max[count, :] = numpy.transpose(U[:, 0])
        input_direction_max[count, :] = numpy.transpose(V[:, 0])

        output_direction_min[count, :] = numpy.transpose(U[:, -1])
        input_direction_min[count, :] = numpy.transpose(V[:, -1])

        store_max[count] = S[0]
        store_min[count] = S[1]

        count = count+1

    #  plot of the singular values , maximum ans minimum
    plt.figure(1)
    plt.subplot(211)
    plt.title('Max and Min Singular values over Freq')
    plt.ylabel('Singular Value')
    plt.xlabel('w')
    plt.loglog(w, store_max, 'r')
    plt.loglog(w, store_min, 'b')

    #  plot of the condition number

    plt.subplot(212)
    plt.title('Condition Number over Freq')
    plt.xlabel('w')
    plt.ylabel('Condition Number')
    plt.loglog(w, store_max/store_min)



    #  plots of different inputs to the maximum ans minimum
    plot_freq_subplot(plt, w, input_direction_max, "max Input", "r.", 2)
    plot_freq_subplot(plt, w, input_direction_min, "min Input", "b.", 2)

    plot_freq_subplot(plt, w, output_direction_max, "max Output", "r.", 3)
    plot_freq_subplot(plt, w, output_direction_min, "min Output", "b.", 3)

    #  plotting of the resulting max and min of the output vectore

if __name__ == '__main__': # only executed when called directly, not executed when imported

    def G(s):
        G = [[1/(s+1), 1/(10*s+1)**2, 1], 
             [0.4/(s*(s+3)), -0.1/(s**2+1), 1],
             [1/(s+1), 10/(s+11), 1/(s+0.001)]]
        return G
    
    def Time_delay(w):
        dead = [[0, 1], [0, 3]]
        return np.exp(dead), dead
    
    SVD_w(G, -3, 3) # TODO remove original SVD_w routines, replace with utilsplots
    
    plt.figure('Input singular vectors')
    sv_dir_plot(G, 'input')
    plt.figure('Output singular vectors')
    sv_dir_plot(G, 'output')
    
    plt.show()
