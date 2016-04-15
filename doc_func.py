from __future__ import print_function, print_function

import numpy
import matplotlib.pyplot as plt


def G(s):
    return 1/(s + 1)


def wI(s):
    return (0.125*s + 0.25)/(0.125*s/4 + 1)


def lI(Gp, G):
    return numpy.abs((Gp - G) / G)


def satisfy(wI, G, Gp, params, s):
    distance = numpy.zeros((len(params), len(s)))
    distance_min = numpy.zeros(len(params))
    for i in range(len(params)):
        for j in range(len(s)):
            distance[i, j] = numpy.abs(wI(s[j])) - lI(Gp(G, params[i], s[j]), G(s[j]))
        distance_min[i] = numpy.min(distance[i, :])
    param_range = params[distance_min > 0]

    return param_range


def plot_range(G, Gprime, wI, w):
    s = 1j*w
    for part, params, G_func, min_max, label in Gprime:
        param_range = satisfy(wI, G, G_func, params, s)
        param_max = numpy.max(param_range)
        param_min = numpy.min(param_range)

        plt.figure()
        plt.loglog(w, numpy.abs(wI(s)), label='$w_{I}$')
        plt.loglog(w, lI(G_func(G,param_max, s), G(s)), label=label)

        if min_max:
            print(part + ' ' + str(param_min) + ' to ' + str(param_max))
            plt.loglog(w, lI(G_func(G, param_min, s), G(s)), label=label)
        else:
            print(part + ' ' + str(param_max))

        plt.xlabel('Frequency  [rad/s]')
        plt.ylabel('Magnitude')
        plt.legend(loc='best')
    plt.show()


def Gp_a(G, theta, s):
    return G(s) * numpy.exp(-theta * s)


def Gp_b(G, tau, s):
    return G(s)/(tau*s + 1)


def Gp_c(G, a, s):
    return 1/(s + a)


def Gp_d(G, T, s):
    return 1/(T*s + 1)


def Gp_e(G, zeta, s):
    return G(s)/((s/70)**2 + 2*zeta*(s/10) + 1)


def Gp_f(G, m, s):
    return G(s)*(1/(0.01*s + 1))**m


def Gp_g(G, tauz, s):
    return G(s)*(-tauz*s + 1)/(tauz*s + 1)

w_start = w_end = points = None


def frequency_plot_setup(axlim, w_start=None, w_end=None, points=None):

    if axlim is None:
        axlim = [None, None, None, None]
    plt.gcf().set_facecolor('white')
    if w_start:
        w = numpy.logspace(w_start, w_end, points)
        s = w*1j
        return s, w
