import numpy as np
import scipy as sc
import scipy.signal as scs
import matplotlib.pyplot as plt


def G(w):
    """ function to create the matrix of transfer functions"""
    s = w*1j

    # the matrix transfer function
    G = [[1/(s+1), 1/(10*s+1)**2, 1], [0.4/((s)*(s+3)), -0.1/(s**2+1), 1], [1/(s+1), 10/(s+11), 1/(s+0.001)]]
    return G


def Time_delay(w):
    dead = [[0, 1], [0, 3]]
    return np.exp(dead), dead


def SVD_w(w_start, w_end):
    """ singular value demoposition
    freqeuncy dependant SVD of G
    w_start = start of the logspace for the freqeuncy range"""

    #  this is an open loop study

    #  w_end = end of the logspace for the frequency range

    w = np.logspace(w_start, w_end, 10000)

    #  getting the size of the system
    A = G(0.0001)
    [U, s, V] = np.linalg.svd(A)
    output_direction_max = np.zeros([U.shape[0], len(w)])
    input_direction_max = np.zeros([V.shape[0], len(w)])
    output_direction_min = np.zeros([U.shape[0], len(w)])
    input_direction_min = np.zeros([V.shape[0], len(w)])


    store_max = np.zeros(len(w))
    store_min = np.zeros(len(w))
    count = 0
    for w_iter in w:
        A = G(w_iter)
        [U, S, V] = np.linalg.svd(A)

        output_direction_max[:, count] = U[:, 0]
        input_direction_max[:, count] = V[:, 0]

        output_direction_min[:, count] = U[:, -1]
        input_direction_min[:, count] = V[:, -1]

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

    def plot_direction(direction, name, color):
        if (direction.shape[0])>2:
            for i in range(direction.shape[0]):
                label = '%s Input Dir %i' % (name, i+1)
                plt.xlabel(label)

                plt.subplot((direction.shape[0]), 1, i + 1)
                plt.semilogx(w, direction[i, :], color)

        else:
            plt.subplot(211)
            plt.semilogx(w, direction[0, :], color)
            plt.subplot(212)
            plt.semilogx(w, direction[1, :], color)

    plt.figure(2)
    plot_direction(input_direction_max, "max Input", "r.")
    plot_direction(input_direction_min, "min Input", "b.")
    plt.figure(3)
    plot_direction(output_direction_max, "max Output", "r.")
    plot_direction(output_direction_min, "min Output", "b.")

    #  plotting of the resulting max and min of the output vectore
    plt.show()

SVD_w(-3, 3)
