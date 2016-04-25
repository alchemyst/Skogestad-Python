# -*- coding: utf-8 -*-
"""
Common features to plotting functions in this script

Default parameters
------------------
axlim : list [xmin, xmax, ymin, ymax]
        A list containing the minimum and maximum limits for the x and y-axis.
        To autoscale a limit enter 'None' in its placeholder.
        The default is to allow autoscaling of the axes.

w_start : float
          The x-axis valve at which to start the plot.

w_end : float
        The x-axis value at which to stop plotting.

points : float
         The number of data points to be used in generating the plot.

Example
-------
def G(s):
    return numpy.matrix([[s/(s+1), 1],
                         [s**2 + 1, 1/(s+1)]])

plt.figure('Example 1')
your_utilsplot_functionA(G, w_start=-5, w_end=2, axlim=[None, None, 0, 1], more_parameters)
plt.show()

plt.figure('Example 2')
plt.subplot(2, 1, 1)
your_utilsplot_functionB(G)
plt.subplot(2, 1, 2)
your_utilsplot_functionC(G)
plt.show()

"""
from __future__ import print_function
import numpy #do not abbreviate this module as np in utilsplot.py
import matplotlib.pyplot as plt
import utils
import doc_func as df


def adjust_spine(xlabel, ylabel, x0=0, y0=0, width=1, height=1):
    """
    General function to adjust the margins for subplots.

    Parameters
    ----------
    xlabel : string
        Label on the main x-axis.
    ylabel : string
        Label on the main x-axis.
    x0 : integer
        Horizontal offset of xlabel.
    y0 : integer
        Verticle offset of ylabel.
    width : float
        Scaling factor of width of subplots.
    height : float
        Scaling factor of height of subplots.

    Returns
    -------
    fig : matplotlib subplot area
    """

    f = plt.get_fignums()[-1]  # useful when multiple figures are added
    fig = plt.figure(f)  # call of plt.figure('plot name') still required externally
    bigax = fig.add_subplot(111)
    bigax.spines['top'].set_color('none')  # remove solid line on major axis
    bigax.spines['bottom'].set_color('none')
    bigax.spines['left'].set_color('none')
    bigax.spines['right'].set_color('none')
    bigax.tick_params(labelcolor='grey', top='off', bottom='off',
                      left='off', right='off')  # remove dashes on major axis
    plt.setp(bigax.get_xticklabels(), visible=False)  # remove values on major axis
    plt.setp(bigax.get_yticklabels(), visible=False)

    box = bigax.get_position()
    bigax.set_position([box.x0 + x0, box.y0 + y0,
                        box.width * width, box.height * height])
    bigax.set_xlabel(xlabel)
    bigax.set_ylabel(ylabel)
    return fig


def plot_freq_subplot(plt, w, direction, name, color, figure_num):
    plt.figure(figure_num)
    N = direction.shape[0]
    for i in range(N):
        #label = '%s Input Dir %i' % (name, i+1)

        plt.subplot(N, 1, i + 1)
        plt.title(name)
        plt.semilogx(w, direction[i, :], color)


###############################################################################
#                                Chapter 2                                    #
###############################################################################


def bode(G, w_start=-2, w_end=2, axlim=None, points=1000, margin=False):
    """
    Shows the bode plot for a plant model

    Parameters
    ----------
    G : tf
        Plant transfer function.
    margin : boolean
        Show the cross over frequencies on the plot (optional).

    Returns
    -------
    GM : array containing a real number
        Gain margin.
    PM : array containing a real number
        Phase margin.
    Plot : matplotlib figure
    """

    s, w, axlim = df.frequency_plot_setup(axlim, w_start, w_end, points)
    plt.clf()

    GM, PM, wc, w_180 = utils.margins(G)

    # plotting of Bode plot and with corresponding frequencies for PM and GM
#    if ((w2 < numpy.log(w_180)) and margin):
#        w2 = numpy.log(w_180)

    # Magnitude of G(jw)
    plt.subplot(2, 1, 1)
    gains = numpy.abs(G(s))
    plt.loglog(w, gains)
    if margin:
        plt.axvline(w_180, color='black')
        plt.text(w_180, numpy.average([numpy.max(gains), numpy.min(gains)]), r'$\angle$G(jw) = -180$\degree$')
    plt.axhline(1., color='red', linestyle='--')
    plt.axis(axlim)
    plt.grid()
    plt.ylabel('Magnitude')

    # Phase of G(jw)
    plt.subplot(2, 1, 2)
    phaseangle = utils.phase(G(s), deg=True)
    plt.semilogx(w, phaseangle)
    if margin:
        plt.axvline(wc, color='black')
        plt.text(wc, numpy.average([numpy.max(phaseangle), numpy.min(phaseangle)]), '|G(jw)| = 1')
    plt.axhline(-180., color='red', linestyle='--')
    plt.axis(axlim)
    plt.grid()
    plt.ylabel('Phase')
    plt.xlabel('Frequency [rad/unit time]')

    return GM, PM


def bodeclosedloop(G, K, w_start=-2, w_end=2, axlim=None, points=1000, margin=False):
    """
    Shows the bode plot for a controller model

    Parameters
    ----------
    G : tf
        Plant transfer function.
    K : tf
        Controller transfer function.
    margin : boolean
        Show the cross over frequencies on the plot (optional).
    """

    _, w, axlim = df.frequency_plot_setup(axlim, w_start, w_end, points)

    L = G(1j*w) * K(1j*w)
    S = utils.feedback(1, L)
    T = utils.feedback(L, 1)

    plt.subplot(2, 1, 1)
    plt.loglog(w, abs(L))
    plt.loglog(w, abs(S))
    plt.loglog(w, abs(T))
    plt.axis(axlim)
    plt.grid()
    plt.ylabel("Magnitude")
    plt.legend(["L", "S", "T"])

    if margin:
        plt.plot(w, 1/numpy.sqrt(2) * numpy.ones(len(w)), linestyle='dotted')

    plt.subplot(2, 1, 2)
    plt.semilogx(w, utils.phase(L, deg=True))
    plt.semilogx(w, utils.phase(S, deg=True))
    plt.semilogx(w, utils.phase(T, deg=True))
    plt.axis(axlim)
    plt.grid()
    plt.ylabel("Phase")
    plt.xlabel("Frequency [rad/s]")


###############################################################################
#                                Chapter 4                                    #
###############################################################################


def mimo_bode(G, w_start=-2, w_end=2, axlim=None, points=1000, Kin=None, text=False, sv_all=False):
    """
    Plots the max and min singular values of G and computes the crossover
    frequency.

    If a controller is specified, the max and min singular values of S are also
    plotted and the bandwidth frequency computed (p81).

    Parameters
    ----------
    G : numpy matrix
        Matrix of plant transfer functions.
    Kin : numpy matrix
        Controller matrix (optional).
    text : boolean
        If true, the crossover and bandwidth frequencies are plotted (optional).
    sv_all : boolean
        If true, plot all the singular values of the plant (optional).

    Returns
    -------
    wC : real
        Crossover frequency.

    wB : real
        Bandwidth frequency.

    Plot : matplotlib figure

    Example
    -------
    >>> K = numpy.array([[1., 2.],
    ...                  [3., 4.]])*10
    >>> t1 = numpy.array([[5., 5.],
    ...                   [5., 5.]])
    >>> t2 = numpy.array([[5., 6.],
    ...                   [7., 8.]])
    >>>
    >>> def G(s):
    ...     return K*numpy.exp(-t1*s)/(t2*s + 1.)
    >>>
    >>> def Kc(s):
    ...     return numpy.array([[0.1, 0.],
    ...                         [0., 0.1]])*10.
    >>> mimo_bode(G, -3, 3, Kc)
    Bandwidth is a tuple of wC, wB
    (0.55557762223988783, 1.3650078065460138)

    """

    s, w, axlim = df.frequency_plot_setup(axlim, w_start, w_end, points)

    if Kin is not None:
        plt.subplot(2, 1, 1)

    dim = numpy.shape(G(0.00001))[0]

    def subbode(text, crossover, labB, labP):
        Sv = numpy.zeros((len(w), dim), dtype=complex)
        f = False
        wA = 0
        for i in range(len(w)):
            Sv[i, :] = utils.sigmas(G(s[i]))
            if not f:
                if (labB == 'wC' and Sv[i, -1] < 1) or (labB == 'wB' and Sv[i, 0] > 0.707):
                    wA = w[i]
                    f = True
        ymin = numpy.min(Sv[:, -1])

        if not sv_all:
            plt.loglog(w, Sv[:, 0], 'b', label=('$\sigma_{max}(%s)$') % labP)
            plt.loglog(w, Sv[:, -1], 'g', label=('$\sigma_{min}(%s)$') % labP)
        else:
            for j in range(dim):
                plt.loglog(w, Sv[:, j], label=('$\sigma_{%s}(%s)$' % (j, labP)))
        plt.axhline(crossover, ls=':', lw=2, color='r')
        if text:
            plt.axvline(wA, ls=':', lw=2, color='r')
            plt.text(wA*1.1, ymin*1.1, labB, color='r')
        plt.axis(axlim)
        plt.grid()
        plt.xlabel('Frequency [rad/unit time]')
        plt.ylabel('$\sigma$')
        plt.legend()
        return wA

    wC = subbode(G, text, 1, 'wC', 'G')

    if Kin is None:
        Bandwidth = wC
        if text:
            print('wC = {:.3}'.format(wC))
    else:
        L = Kin(s) * G(s)
        S = numpy.linalg.inv(numpy.eye(dim) + L)  # SVD of S = 1/(I + L)

        wB = subbode(S, text, 0.707, 'wC', 'G')

        Bandwidth = wC, wB
        if text:
            print('(wC = {1}, wB = {2}'.format(wC, wB))

    return Bandwidth


def mino_nyquist_plot(L, w_start=-2, w_end=2, axlim=None, points=1000):
    """
    Nyquist stability plot for MIMO system.

    Parameters
    ----------
    L : numpy matrix
        Closed loop transfer function matrix as a function of s, i.e. def L(s).

    Returns
    -------
    Plot : matplotlib figure

    Example
    -------
    >>> K = numpy.array([[1., 2.],
    ...                  [3., 4.]])
    >>> t1 = numpy.array([[5., 5.],
    ...                   [5., 5.]])
    >>> t2 = numpy.array([[5., 6.],
    ...                   [7., 8.]])
    >>> Kc = numpy.array([[0.1, 0.],
    ...                   [0., 0.1]])*6
    >>>
    >>> def G(s):
    ...     return K*numpy.exp(-t1*s)/(t2*s + 1)
    ...
    >>> def L(s):
    ...     return Kc*G(s)
    ...
    >>> #MIMOnyqPlot(L, 2)

    """

    _, w, axlim = df.frequency_plot_setup(axlim, w_start, w_end, points)

    Lin = numpy.zeros((len(w)), dtype=complex)
    x = numpy.zeros((len(w)))
    y = numpy.zeros((len(w)))
    dim = numpy.shape(L(0.1))
    for i in range(len(w)):
        Lin[i] = numpy.linalg.det(numpy.eye(dim[0]) + L(w[i]*1j))
        x[i] = numpy.real(Lin[i])
        y[i] = numpy.imag(Lin[i])
    plt.plot(x, y, 'k-', lw=1)
    plt.axis(axlim)
    plt.xlabel('Re G(wj)')
    plt.ylabel('Im G(wj)')

    # plotting a unit circle
    x = numpy.linspace(-1, 1, 200)
    y_up = numpy.sqrt(1 - x**2)
    y_down = -1*numpy.sqrt(1 - x**2)
    plt.plot(x, y_up, 'b:', x, y_down, 'b:', lw=2)
    plt.plot(0, 0, 'r*', ms=10)
    plt.grid(True)
    plt.axis('equal')   # Ensure the unit circle remains round on resizing the figure


def sv_dir_plot(G, plot_type, w_start=-2, w_end=2, axlim=None, points=1000):
    """
    Plot the input and output singular vectors associated with the minimum and
    maximum singular values.

    Parameters
    ----------
    G : matrix
        Plant model or sensitivity function.
    plot_type : string
        Type of plot.

        =========      ============================
        plot_type      Type of plot
        =========      ============================
        input          Plots input vectors
        output         Plots output vectors
        =========      ============================

    Returns
    -------
    Plot : matplotlib figure

    Note
    ----
    Can be used with the plant matrix G and the sensitivity function S
    for controlability analysis
    """

    s, w, axlim = df.frequency_plot_setup(axlim, w_start, w_end, points)

    freqresp = [G(si) for si in s]

    if plot_type == 'input':
        vec = numpy.array([V for _, _, V in map(utils.SVD, freqresp)])
        d = 'v'
    elif plot_type == 'output':
        vec = numpy.array([U for U, _, _ in map(utils.SVD, freqresp)])
        d = 'u'
    else:
        raise ValueError('Invalid plot_type parameter.')

    dim = numpy.shape(vec)[1]
    for i in range(dim):
        plt.subplot(dim, 1, i + 1)
        plt.semilogx(w, vec[:, 0, i], label= '$%s_{max}$' % d, lw=4)
        plt.semilogx(w, vec[:, -1, i], label= '$%s_{min}$' % d, lw=4)
        plt.axhline(0, color='red', ls=':')
        plt.axis(axlim)
        plt.ylabel('$%s_%s$' % (d, i + 1))
        plt.legend()

    plt.xlabel('Frequency [rad/unit time]')


###############################################################################
#                                Chapter 6                                    #
###############################################################################


def condtn_nm_plot(G, w_start=-2, w_end=2, axlim=None, points=1000):
    '''
    Plot of the condition number, the maximum over the minimum singular value

    Parameters
    ----------
    G : numpy matrix
        Plant model.

    Returns
    -------
    Plot : matplotlib figure

    Note
    ----
    A condition number over 10 may indicate sensitivity to uncertainty and
    control problems

    With a smallcondition number, Gamma =1, the system is insensitive to
    inputuncertainty, irrespective of controller (p248).
    '''

    s, w, axlim = df.frequency_plot_setup(axlim, w_start, w_end, points)

    def cndtn_nm(G):
        return utils.sigmas(G)[0]/utils.sigmas(G)[-1]

    freqresp = [G(si) for si in s]

    plt.loglog(w, [cndtn_nm(Gfr) for Gfr in freqresp], label='$\gamma (G)$')
    plt.axis(axlim)
    plt.ylabel('$\gamma (G)$')
    plt.xlabel('Frequency [rad/unit time]')
    plt.axhline(10., color='red', ls=':', label='"Large" $\gamma (G) = 10$')
    plt.legend()


def rga_plot(G, w_start=-2, w_end=2, axlim=None, points=1000, fig=0, plot_type='elements', input_label=None, output_label=None):
    '''
    Plots the relative gain interaction between each output and input pairing

    Parameters
    ----------
    G : numpy matrix
        Plant model.
    plot_type : string
        Type of plot.

        =========      ============================
        plot_type      Type of plot
        =========      ============================
        all            All the RGAs on one plot
        output         Plots grouped by output
        input          Plots grouped by input
        element        Each element has its own plot
        =========      ============================

    Returns
    -------
    Plot : matplotlib figure

    Example
    -------
    Adapted from example 3.11 pg 86 S. Skogestad

    >>> def G(s):
    ...     G = 0.01**(-5*s)/((s + 1.72e-4)*(4.32*s + 1))*numpy.matrix([[-34.54*(s + 0.0572), 1.913], [-30.22*s, -9.188*(s + 6.95e-4)]])
    ...     return G
    >>> rga_plot(G, w_start=-5, w_end=2, axlim=[None, None, 0., 1.])
    '''

    s, w, axlim = df.frequency_plot_setup(axlim, w_start, w_end, points)

    dim = G(0).shape # Number of rows and columns in SS transfer function
    freqresp = [G(si) for si in s]

    plot_No = 1

    if (input_label is None) and (output_label is None):
        labels = False
    elif numpy.shape(input_label)[0] == numpy.shape(output_label)[0]:
        labels = True
    else:
        raise ValueError('Input and output label count is not equal')

    if plot_type == 'elements':
        fig = adjust_spine('Frequency [rad/unit time]','RGA magnitude', -0.05, -0.03, 0.8, 0.9)
        for i in range(dim[0]):
            for j in range(dim[1]):
                ax = fig.add_subplot(dim[0], dim[1], plot_No)
                if labels:
                    ax.set_title('Output (%s) vs. Input (%s)' % (output_label[i], input_label[j]))
                else:
                    ax.set_title('Output %s vs. Input %s' % (i + 1, j + 1))
                ax.semilogx(w, numpy.array(numpy.abs(([utils.RGA(Gfr)[i, j] for Gfr in freqresp]))))
                plot_No += 1

                ax.axis(axlim)
                ax.set_ylabel('$|\lambda$$_{%s, %s}|$' % (i + 1, j + 1))
                box = ax.get_position()
                ax.set_position([box.x0, box.y0,
                                 box.width * 0.8, box.height * 0.9])

    elif plot_type == 'outputs': #i
        fig = adjust_spine('Frequency [rad/unit time]','RGA magnitude', -0.05, -0.03, 1, 0.9)
        for i in range(dim[0]):
            ax = fig.add_subplot(dim[1], 1, plot_No)
            ax.set_title('Output %s vs. input j' % (i + 1))
            rgamax = []
            for j in range(dim[1]):
                rgas = numpy.array(numpy.abs(([utils.RGA(Gfr)[i, j] for Gfr in freqresp])))
                ax.semilogx(w, rgas, label='$\lambda$$_{%s, %s}$' % (i + 1, j + 1))
                rgamax.append(max(rgas))

                if j == dim[1] - 1: #self-scaling algorithm
                    if axlim is not None:
                        ax.axis(axlim)
                    else:
                        ax.axis([None, None, None, max(rgamax)])

            ax.set_ylabel('$|\lambda$$_{%s, j}|$' % (i + 1))
            box = ax.get_position()
            ax.set_position([box.x0, box.y0,
                             box.width, box.height * 0.9])
            ax.legend()
            plot_No += 1

    elif plot_type == 'inputs': #j
        fig = adjust_spine('Frequency [rad/unit time]','RGA magnitude', -0.05, -0.03, 1, 0.9)
        for j in range(dim[1]):
            ax = fig.add_subplot(dim[0], 1, plot_No)
            ax.set_title('Output i vs. input %s' % (j + 1))
            rgamax = []
            for i in range(dim[0]):
                rgas = numpy.array(numpy.abs(([utils.RGA(Gfr)[i, j] for Gfr in freqresp])))
                ax.semilogx(w, rgas, label='$\lambda$$_{%s, %s}$' % (i + 1, j + 1))
                rgamax.append(max(rgas))

                if i == dim[1] - 1: #self-scaling algorithm
                    if axlim is not None:
                        ax.axis(axlim)
                    else:
                        ax.axis([None, None, None, max(rgamax)])

            ax.set_ylabel('$|\lambda$$_{i, %s}|$' % (j + 1))
            box = ax.get_position()
            ax.set_position([box.x0, box.y0,
                             box.width, box.height * 0.9])
            ax.legend()
            plot_No += 1

    elif plot_type == 'all':
        for i in range(dim[0]):
            for j in range(dim[1]):
                plt.semilogx(w, numpy.array(numpy.abs(([utils.RGA(Gfr)[i, j] for Gfr in freqresp]))))
                plt.axis(axlim)
                plt.ylabel('|$\lambda$$_{i,j}$|')
                plt.xlabel('Frequency [rad/unit time]')

    else:
        raise ValueError("Invalid plot_type paramter.")


def rga_nm_plot(G, pairing_list=None, pairing_names=None, w_start=-2, w_end=2, axlim=None, points=1000, plot_type='all'):
    '''
    Plots the RGA number for a specified pairing

    Parameters
    ----------
    G : numpy matrix
        Plant model.
    pairing_list : List of sparse numpy matrixes of the same shape as G.
        An array of zeros with a 1. at each required output-input pairing.
        The default is a diagonal pairing with 1.'s on the diagonal.
    plot_type : string
        Type of plot:

        =========      =============================
        plot_type      Type of plot
        =========      =============================
        all            All the pairings on one plot
        element        Each pairing has its own plot
        =========      =============================

    Returns
    -------
    Plot : matplotlib figure

    Example
    -------
    # Adapted from example 3.11 pg 86 S. Skogestad
    >>> def G(s):
    ...     G = 0.01**(-5*s)/((s + 1.72e-4)*(4.32*s + 1))*numpy.matrix([[-34.54*(s + 0.0572), 1.913], [-30.22*s, -9.188*(s + 6.95e-4)]])
    ...     return G
    >>> pairing = numpy.matrix([[1., 0.], [0., 1.]])
    >>> rga_nm_plot(G, [pairing], w_start=-5, w_end=2, axlim=[None, None, 0., 1.])

    Note
    ----
    This plotting function can only be used on square systems
    '''

    s, w, axlim = df.frequency_plot_setup(axlim, w_start, w_end, points)

    dim = numpy.shape(G(0)) # Number of rows and columns in SS transfer function
    freqresp = [G(si) for si in s]

    if pairing_list is None: # Setting a blank entry to the default of a diagonal comparison
        pairing_list = numpy.identity(dim[0])
        pairing_names ='Diagonal pairing'
    else:
        for pairing in pairing_list:
            if pairing.shape != dim:
                raise ValueError('RGA_Number_Plot on plots square n by n matrices, make sure input matrix is square')

    plot_No = 0

    if plot_type == 'all':
        for p in pairing_list:
            plt.semilogx(w, [utils.RGAnumber(Gfr, p) for Gfr in freqresp], label=pairing_names[plot_No])
            plot_No += 1
        plt.axis(axlim)
        plt.ylabel('||$\Lambda$(G) - I||$_{sum}$')
        plt.xlabel('Frequency [rad/unit time]')
        plt.legend()

    elif plot_type == 'element':
        pcount = numpy.shape(pairing_list)[0] # pairing_list.count not accessible
        for p in pairing_list:
            plot_No += 1
            plt.subplot(1, pcount, plot_No)
            plt.semilogx(w, [utils.RGAnumber(Gfr, p) for Gfr in freqresp])
            plt.axis(axlim)
            plt.title(pairing_names[plot_No - 1])
            plt.xlabel('Frequency [rad/unit time]')
            if plot_No == 1:
                plt.ylabel('||$\Lambda$(G) - I||$_{sum}$')
    else:
        raise ValueError("Invalid plot_type paramter.")


def dis_rejctn_plot(G, Gd, S=None, w_start=-2, w_end=2, axlim=None, points=1000):
    '''
    A subplot of disturbance conditition number to check for input saturation
    (equation 6.43, p238). Two more subplots indicate if the disturbances fall
    withing the bounds of S, applying equations 6.45 and 6.46 (p239).

    Parameters
    ----------
    G : numpy matrix
        Plant model.
    Gd : numpy matrix
        Plant disturbance model.
    S : numpy matrix
        Sensitivity function (optional, if available).
    # TODO test S condition
    Returns
    -------
    Plot : matplotlib figure

    Note
    ----
    The disturbance condition number provides a measure of how a disturbance gd
    is aligned with the plant G. Alignment can vary between 1 and the condition
    number.

    For acceptable performance the singular values of S must fall below the
    inverse 2-norm of gd.
    '''

    s, w, axlim = df.frequency_plot_setup(axlim, w_start, w_end, points)

    dim = numpy.shape(Gd(0))[1] # column count
    inv_norm_gd = numpy.zeros((dim, points))
    yd = numpy.zeros((dim, points, numpy.shape(Gd(0))[0]), dtype=complex) # row count
    condtn_nm_gd = numpy.zeros((dim, points))
    for i in range(dim):
        for k in range(points):
            inv_norm_gd[i, k], yd[i, k, :], condtn_nm_gd[i, k] = utils.distRej(G(s[k]), Gd(s[k])[:, i])

    if S is None: sub = 2
    else: sub = 3

    # Equation 6.43
    plt.subplot(sub, 1, 1)
    for i in range(dim):
        plt.loglog(w, condtn_nm_gd[i], label=('$\gamma_{d%s} (G)$' % (i+1)))
    plt.axhline(1., color='red', ls=':')
    plt.axis(axlim)
    plt.xlabel('Frequency [rad/unit time]')
    plt.ylabel('$\gamma$$_d (G)$')
    plt.axhline(1., color='red', ls=':')
    plt.legend()

    # Equation 6.44
    plt.subplot(sub, 1, 2)
    for i in range(dim):
        plt.loglog(w, inv_norm_gd[i], label=('$1/||g_{d%s}||_2$' % (i+1)))
        if not S is None:
            S_yd = numpy.array([numpy.linalg.norm(S(p) * yd[i, p, :].T, 2) for p in range(points)])
            plt.loglog(w, S_yd, label='$||Sy_d||_2$')
    plt.axis(axlim)
    plt.xlabel('Frequency [rad/unit time]')
    plt.legend()

    if not S is None: # this subplot should not be repeated with S is not avaiable
        # Equation 6.45
        plt.subplot(3, 1, 3)
        for i in range(dim):
            plt.loglog(w, inv_norm_gd[i], label=('$1/||g_{d%s}||_2$' % (i+1)))
            s_min = numpy.array([utils.sigmas(S(s[p]), 'min') for p in range(points)])
            s_max = numpy.array([utils.sigmas(S(s[p]), 'max') for p in range(points)])
            plt.loglog(w, s_min, label='$\sigma_{min}$')
            plt.loglog(w, s_max, label='$\sigma_{max}$')
        plt.axis(axlim)
        plt.xlabel('Frequency [rad/unit time]')
        plt.legend()


def input_perfect_const_plot(G, Gd, w_start=-2, w_end=2, axlim=None, points=1000, simultaneous=False):
    '''
    Plot for input constraints for perfect control. Applies equation 6.50
    (p240).

    Parameters
    ----------
    G : numpy matrix
        Plant model.
    Gd : numpy matrix
        Plant disturbance model.
    simultaneous : boolean.
        If true, the induced max-norm is calculated for simultaneous
        disturbances (optional).

    Returns
    -------
    Plot : matplotlib figure

    Note
    ----
    The boundary conditions is values below 1 (p240).
    '''

    s, w, axlim = df.frequency_plot_setup(axlim, w_start, w_end, points)

    dim = numpy.shape(Gd(0))[1]
    perfect_control = numpy.zeros((dim, points))
    #if simultaneous: imn = np.zeros(points)
    for i in range(dim):
        for k in range(points):
            Ginv = numpy.linalg.inv(G(s[k]))
            perfect_control[i, k] = numpy.max(numpy.abs(Ginv * Gd(s[k])[:, i]))
            #if simultaneous:
            #TODO induced max-norm
        plt.loglog(w, perfect_control[i], label=('$g_{d%s}$' % (i + 1)))

    plt.axhline(1., color='red', ls=':')
    plt.ylabel(r'$||G^{-1}.g_d||_{max}$')
    plt.xlabel('Frequency [rad/unit time]')
    plt.grid(True)
    plt.axis(axlim)
    plt.legend()


def ref_perfect_const_plot(G, R, wr, w_start=-2, w_end=2, axlim=None, points=1000, plot_type='all'):
    '''
    Use these plots to determine the constraints for perfect control in terms
    of combined reference changes. Equation 6.52 (p241) calculates the
    minimal requirement for input saturation to check in terms of set point
    tracking. A more tighter bounds is calculated with equation 6.53 (p241).

    Parameters
    ----------
    G : tf
        Plant transfer function.
    R : numpy matrix (n x n)
        Reference changes (usually diagonal with all elements larger than 1)
    wr : float
        Frequency up to witch reference tracking is required
    type_eq : string
        Type of plot:

        =========      ==================================
        plot_type      Type of plot
        =========      ==================================
        minimal        Minimal requirement, equation 6.52
        tighter        Tighter requirement, equation 6.53
        allo           All requirements
        =========      ==================================

    Returns
    -------
    Plot : matplotlib figure

    Note
    ----
    All the plots in this function needs to be larger than 1 for perfect
    control, otherwise input saturation will occur.
    '''

    s, w, axlim = df.frequency_plot_setup(axlim, w_start, w_end, points)

    lab1 = '$\sigma_{min} (G(jw))$'
    bound1 = [utils.sigmas(G(si), 'min') for si in s]
    lab2 = '$\sigma_{min} (R^{-1}G(jw))$'
    bound2 = [utils.sigmas(numpy.linalg.pinv(R) * G(si), 'min') for si in s]

    if plot_type == 'all':
        plt.loglog(w, bound1, label=lab1)
        plt.loglog(w, bound2, label=lab2)
    elif plot_type == 'minimal':
        plt.loglog(w, bound1, label=lab1)
    elif plot_type == 'tighter':
        plt.loglog(w, bound2, label=lab2)

    else: raise ValueError('Invalid plot_type parameter.')

    mm = bound1 + bound2 + [1] # ensures that whole graph is visible
    plt.loglog([wr, wr], [0.5 * numpy.min(mm), 5 * numpy.max(mm)], 'r', ls=':', label='Ref tracked')
    plt.loglog([w[0], w[-1]], [1, 1], 'r', label='Bound')
    plt.legend()


def input_acceptable_const_plot(G, Gd, w_start=-2, w_end=2, axlim=None, points=1000, modified=False):
    '''
    Subbplots for input constraints for accepatable control. Applies equation
    6.55 (p241).

    Parameters
    ----------
    G : numpy matrix
        Plant model.
    Gd : numpy matrix
        Plant disturbance model.
    modified : boolean
        If true, the arguments in the equation are change to :math:`\sigma_1
        (G) + 1 \geq |u_i^H g_d|`. This is to avoid a negative log scale.

    Returns
    -------
    Plot : matplotlib figure

    Note
    ----
    This condition only holds for :math:`|u_i^H g_d|>1`.
    '''

    s, w, axlim = df.frequency_plot_setup(axlim, w_start, w_end, points)

    freqresp = [G(si) for si in s]
    sig = numpy.array([utils.sigmas(Gfr) for Gfr in freqresp])
    one = numpy.ones(points)

    plot_No = 1

    dimGd = numpy.shape(Gd(0))[1]
    dimG = numpy.shape(G(0))[0]
    acceptable_control = numpy.zeros((dimGd, dimG, points))
    for j in range(dimGd):
        for i in range(dimG):
            for k in range(points):
                U, _, _ = utils.SVD(G(s[k]))
                acceptable_control[j, i, k] = numpy.abs(U[:, i].H * Gd(s[k])[:, j])
            plt.subplot(dimG, dimGd, plot_No)
            if not modified:
                plt.loglog(w, sig[:, i], label=('$\sigma_%s$' % (i + 1)))
                plt.plot(w, acceptable_control[j, i] - one, label=('$|u_%s^H.g_{d%s}|-1$' % (i + 1, j + 1)))
            else:
                plt.loglog(w, sig[:, i] + one, label=('$\sigma_%s+1$' % (i + 1)))
                plt.plot(w, acceptable_control[j, i], label=('$|u_%s^H.g_{d%s}|$' % (i + 1, j + 1)))
                plt.loglog([w[0], w[-1]], [1, 1], 'r', ls=':', label='Applicable')
            plt.xlabel('Frequency [rad/unit time]')
            plt.grid(True)
            plt.axis(axlim)
            plt.legend()
            plot_No += 1


###############################################################################
#                                Chapter 2 (move to top)                      #
###############################################################################


def step(G, t_end=100, initial_val=0, input_label=None, output_label=None, points=1000):
    '''
    This function is similar to the MatLab step function.

    Parameters
    ----------
    G : tf
        Plant transfer function.
    t_end : integer
        Time period which the step response should occur (optional).
    initial_val : integer
        Starting value to evalaute step response (optional).
    input_label : array
        List of input variable labels.
    output_label : array
        List of output variable labels.

    Returns
    -------
    Plot : matplotlib figure
    '''

    plt.gcf().set_facecolor('white')

    rows = numpy.shape(G(0))[0]
    columns = numpy.shape(G(0))[1]

    s = utils.tf([1, 0])
    system = G(s)

    if ((input_label is None) and (input_label is None)):
        labels = False
    elif (numpy.shape(input_label)[0] == columns) and (numpy.shape(output_label)[0] == rows):
        labels = True
    else:
        raise ValueError('Label count is inconsistent to plant size')

    fig = adjust_spine('Time','Output magnitude', -0.05, 0.1, 0.8, 0.9)

    cnt = 0
    tspace = numpy.linspace(0, t_end, points)
    for i in range(rows):
        for j in range(columns):
            cnt += 1
            nulspace = numpy.zeros(points)
            ax = fig.add_subplot(rows + 1, columns, cnt)
            tf = system[i, j]
            if all(tf.numerator) != 0:
                realstep = numpy.real(tf.step(initial_val, tspace))
                ax.plot(realstep[0], realstep[1])
            else:
                ax.plot(tspace, nulspace)
            if labels:
                ax.set_title('Output (%s) vs. Input (%s)' % (output_label[i], input_label[j]))
            else:
                ax.set_title('Output %s vs. Input %s' % (i + 1, j + 1))

            if i == 0:
                xax = ax
            else:
                ax.sharex = xax

            if j == 0:
                yax = ax
                ax.set_ylabel = output_label[j]
            else:
                ax.sharey =yax

            plt.setp(ax.get_yticklabels(), fontsize=10)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 * 1.05 - 0.05,
                             box.width * 0.8, box.height * 0.9])


def freq_step_response_plot(G, K, Kc, t_end=50, freqtype='S', w_start=-2, w_end=2, axlim=None, points=1000):
    '''
    A subplot function for both the frequency response and step response for a
    controlled plant

    Parameters
    ----------
    G : tf
        Plant transfer function.
    K : tf
        Controller transfer function.
    Kc : integer
        Controller constant.
    t_end : integer
        Time period which the step response should occur.
    freqtype : string (optional)
        Type of function to plot:

        ========    ==================================
        freqtype    Type of function to plot
        ========    ==================================
        S           Sensitivity function
        T           Complementary sensitivity function
        L           Loop function
        ========    ==================================

    Returns
    -------
    Plot : matplotlib figure

    '''

    _, w, axlim = df.frequency_plot_setup(axlim, w_start, w_end, points)

    plt.subplot(1, 2, 1)

    # Controllers transfer function with various controller gains
    Ks = [(kc * K) for kc in Kc]
    Ts = [utils.feedback(G * Kss, 1) for Kss in Ks]

    if freqtype=='S':
        Fs = [(1 - Tss) for Tss in Ts]
        plt.title('(a) Sensitivity function')
        plt.ylabel('Magnitude $|S|$')
    elif freqtype=='T':
        Fs = Ts
        plt.title('(a) Complementary sensitivity function')
        plt.ylabel('Magnitude $|T|$')
    else: #freqtype=='L'
        Fs = [(G*Kss) for Kss in Ks]
        plt.title('(a) Loop function')
        plt.ylabel('Magnitude $|L|$')


    wi = w * 1j
    i = 0
    for F in Fs:
        plt.loglog(w, abs(F(wi)), label='Kc={d%s}=' % Kc[i])
        i =+ 1
    plt.axis(axlim)
    plt.grid(b=None, which='both', axis='both')
    plt.xlabel('Frequency [rad/unit time]')
    plt.legend(["Kc = %1.2f" % k for k in Kc],loc=4)

    plt.subplot(1, 2, 2)
    plt.title('(b) Response to step in reference')
    tspan = numpy.linspace(0, t_end, points)
    for T in Ts:
        [t, y] = T.step(0, tspan)
        plt.plot(t, y)
    plt.plot(tspan, 0 * numpy.ones(points), ls='--')
    plt.plot(tspan, 1 * numpy.ones(points), ls='--')
    plt.axis(axlim)
    plt.xlabel('Time')
    plt.ylabel('$y(t)$')


def step_response_plot(Y, U, t_end=50, initial_val=0, timedim='sec', axlim=None, points=1000, constraint=None, method='numeric'):
    '''
    A plot of the step response of a transfer function

    Parameters
    ----------
    Y : tf
        Output transfer function.
    U : tf
        Input transfer function.
    t_end : integer
        Time period which the step response should occur (optional).
    initial_val : integer
        Starting value to evalaute step response (optional).
    constraint : float
        The upper limit the step response cannot exceed. is only calculated
        if a value is specified (optional).
    method : ['numeric','analytic']
        The method that is used to calculate a constrainted response. A
        constraint value is required (optional).

    Returns
    -------
    Plot : matplotlib figure

    '''

    axlim = df.frequency_plot_setup(axlim)

    [t,y] = utils.tf_step(Y, t_end, initial_val)
    plt.plot(t,y)

    [t,y] = utils.tf_step(U, t_end, initial_val)
    plt.plot(t,y)

    if constraint is None:
        plt.legend(['$y(t)$','$u(t)$'])
    else:
        [t,y] = utils.tf_step(U, t_end, initial_val, points, constraint, Y, method)
        plt.plot(t,y[0])
        plt.plot(t,y[1])

        plt.legend(['$y(t)$','$u(t)$','$u(t) const$','$y(t) const$']) #con = constraint

    plt.plot([0, t_end], numpy.ones(2),'--')

    plt.axis(axlim)
    plt.xlabel('Time [' + timedim + ']')



def perf_Wp_plot(S, wB_req, maxSSerror, w_start, w_end, axlim=None, points=1000):
    '''
    MIMO sensitivity S and performance weight Wp plotting funtion.

    Parameters
    ----------
    S : numpy array
        Sensitivity transfer function matrix as function of s => S(s)
    wB_req : float
        The design or require bandwidth of the plant in rad/time.
        1/time eg: wB_req = 1/20sec = 0.05rad/s
    maxSSerror : float
        The maximum stead state tracking error required of the plant.
    wStart : float
        Minimum power of w for the frequency range in rad/time.
        eg: for w startig at 10e-3, wStart = -3.
    wEnd : float
        Maximum value of w for the frequency range in rad/time.
        eg: for w ending at 10e3, wStart = 3.

    Returns
    -------
    wB : float
        The actualy plant bandwidth in rad/time given the specified controller
        used to generate the sensitivity matrix S(s).
    Plot : matplotlib figure

    Example
    -------
    >>> K = numpy.array([[1., 2.],
    ...                  [3., 4.]])
    >>> t1 = numpy.array([[5., 5.],
    ...                   [5., 5.]])
    >>> t2 = numpy.array([[5., 6.],
    ...                   [7., 8.]])
    >>> Kc = numpy.array([[0.1, 0.],
    ...                   [0., 0.1]])*10
    >>>
    >>> def G(s):
    ...     return K*numpy.exp(-t1*s)/(t2*s + 1)
    ...
    >>> def L(s):
    ...     return Kc*G(s)
    ...
    >>> def S(s):
    ... # SVD of S = 1/(I + L)
    ...     return numpy.linalg.inv((numpy.eye(2) + L(s)))
    >>> perf_Wp(S, 0.05, 0.2, -3, 1)
    '''

    s, w, axlim = df.frequency_plot_setup(axlim, w_start, w_end, points)

    magPlotS1 = numpy.zeros((len(w)))
    magPlotS3 = numpy.zeros((len(w)))
    Wpi = numpy.zeros((len(w)))
    f = 0                                    #f for flag
    for i in range(len(w)):
        _, Sv, _ = utils.SVD(S(s[i]))
        magPlotS1[i] = Sv[0]
        magPlotS3[i] = Sv[-1]
        if f < 1 and magPlotS1[i] > 0.707:
            wB = w[i]
            f = 1
    for i in range(len(w)):
        Wpi[i] = utils.Wp(wB_req, maxSSerror, s[i])

    plt.subplot(2, 1, 1)
    plt.loglog(w, magPlotS1, 'r-', label='Max $\sigma$(S)')
    plt.loglog(w, 1./Wpi, 'k:', label='|1/W$_P$|', lw=2.)
    plt.axhline(0.707, color='green', ls=':', lw=2, label='|S| = 0.707')
    plt.axvline(wB_req, color='blue', ls=':', lw=2)
    plt.text(wB_req*1.1, 7, 'req wB', color='blue', fontsize=10)
    plt.axvline(wB, color='green')
    plt.text(wB*1.1, 0.12, 'wB = %0.3f rad/s' % wB, color='green', fontsize=10)
    plt.axis(axlim)
    plt.grid(True)
    plt.xlabel('Frequency [rad/unit time]')
    plt.ylabel('Magnitude')
    plt.legend(loc='upper left', fontsize=10, ncol=1)

    plt.subplot(2, 1, 2)
    plt.semilogx(w, magPlotS1*Wpi, 'r-', label='|W$_P$S|')
    plt.axhline(1, color='blue', ls=':', lw=2)
    plt.axvline(wB_req, color='blue', ls=':', lw=2, label='|W$_P$S| = 1')
    plt.text(wB_req*1.1, numpy.max(magPlotS1*Wpi)*0.95, 'req wB', color='blue', fontsize=10)
    plt.axvline(wB, color='green')
    plt.text(wB*1.1, 0.12, 'wB = %0.3f rad/s' % wB, color='green', fontsize=10)
    plt.axis(axlim)
    plt.xlabel('Frequency [rad/unit time]')
    plt.ylabel('Magnitude')
    plt.legend(loc='upper right', fontsize=10, ncol=1)

    return wB
