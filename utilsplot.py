# -*- coding: utf-8 -*-
"""
Created on Sat May 31 20:50:02 2014

@author: bmaas

Inputs
------
The default inputs to a plotting function in this script include:

axlim : list
        A list containing the minimum and maximum limits for the x and y-axis.
        To autoscale a limit enter 'None' in its placeholder.
        The default is to allow autoscaling of the axes.
        
w_start : float
          The x-axis valve at which to start the plot.

w_end : float
        The x-axis value at which to stop plotting.

points : float
         The number of data points to be used in generating the plot.

Plots
-----
bode_plot :
    (Can bring across from utils.py)
    
nyquist_plot:
    (Can bring across from utils.py)
    
sv_plot : Maximum and minimum singular values of a matirix
    (Can remove SVD_over_frequency.py, SVD_Total_Analysis.py and SVD_w.py)
    
condtn_nm_plot : A plot of the condition number for a specified diagonal
    
rga_plot: A plot of the relative gain interactions for a matrix over a given frequency
    (Can remove RGA.py script)

rga_nm_plot: A plot of the RGA number for a given pairing

weightd_sens_plot :
    (Can bring across from utils.py)
    
perf_wt_plot :
    (Can bring across from utils.py)
    
dis_rejctn_plot : A plot of the disturbance condition number and the bounds imposed
    by the singular values.
"""

import numpy
import utils
import matplotlib.pyplot as plt
           

def sv_plot(G, axlim=None, w_start=-2, w_end=2, points=100):
    '''
    Plot of Maximum and minimum singular values of a matirix
    
    Parameters
    ----------
    G : numpy array
        plant model or sensitivity function
              
    Returns
    -------
    fig(Max Min SV) : figure
        A figure of the maximum and minimum singular values of the matrix G
    
    Note
    ----
    Can be used with the plant matrix G and the sensitivity function S
    for controlability analysis
    '''

    if axlim is None:
        axlim = [None, None, None, None]

    w = numpy.logspace(w_start, w_end, points)
    s = w*1j    
    
    freqresp = map(G, s)
    
    plt.figure('Min Max SV')
    plt.clf()
    plt.gcf().set_facecolor('white')
    
    plt.semilogx(w, [utils.sigmas(Gfr)[0] for Gfr in freqresp], label=('$\sigma$$_{MAX}$'), color='blue')
    plt.semilogx(w, [utils.sigmas(Gfr)[-1] for Gfr in freqresp], label=('$\sigma$$_{MIN}$'), color='blue', alpha=0.5)
    plt.xlabel('Frequency (rad/unit time)')
    
    plt.axhline(1., color='red', ls=':')
    plt.legend()  
    plt.show()
    return


def condtn_nm_plot(G, axlim=None, w_start=-2, w_end=2, points=100):
    '''
    Plot of the condition number, the maximum over the minimum singular value
    
    Parameters
    ----------
    G : numpy array
        plant model
              
    Returns
    -------
    fig(Condition number) : figure
        A figure of the Condition number.
    
    Note
    ----
    A condition number over 10 may indicate sensitivity to uncertainty and
    control problems
    '''
    
    if axlim is None:
        axlim = [None, None, None, None]

    w = numpy.logspace(w_start, w_end, points)
    s = w*1j    
    
    def cndtn_nm(G):
        return(utils.sigmas(G)[0]/utils.sigmas(G)[-1])
    
    freqresp = map(G, s)
    
    plt.figure('Condition number')
    plt.clf()
    plt.gcf().set_facecolor('white')
    
    plt.semilogx(w, [cndtn_nm(Gfr) for Gfr in freqresp], label=('$\sigma$$_{MAX}$/$\sigma$$_{MIN}$'))
    plt.axis(axlim)
    plt.ylabel('$\gamma$(G)', fontsize = 15)
    plt.xlabel('Frequency (rad/unit time)')
    plt.axhline(10., color='red', ls=':', label=('$\gamma$(G) > 10'))
    plt.legend()
    plt.show()
    return


def rga_plot(G, axlim=None, w_start=-2, w_end=2, points=100):
    '''
    Plots the relative gain interaction between each output and input pairing
    
    Parameters
    ----------
    G : numpy array
        plant model
              
    Returns
    -------
    fig(RGA) : figure
        A figure of subplots for each interaction between an output and
        an input.
    
    Example
    -------
    # Adapted from example 3.11 pg 86 S. Skogestad
    >>> def G(s):
    ...     G = 0.01*numpy.exp(-5*s)/((s + 1.72e-4)*(4.32*s + 1))*numpy.array([[-34.54*(s + 0.0572), 1.913], [-30.22*s, -9.188*(s + 6.95e-4)]])
    ...     return G
    >>> rga_plot(G, axlim=[None, None, 0., 1.], w_start=-5, w_end=2)
    
    
    Note
    ----
    This entry draws on and improves RGA.py.
    If accepted, then RGA.py could be removed from the repository
    '''

    if axlim is None:
        axlim = [None, None, None, None]

    w = numpy.logspace(w_start, w_end, points)
    s = w*1j
    
    dim = numpy.shape(G(0)) # Number of rows and columns in SS transfer function    
    freqresp = map(G, s)
    
    count = 0 # Arrange the individual RGA responses to be compatible with the plotting order of plt.subplot
    rga = numpy.zeros([dim[0]*dim[1], points])
    for i in range(dim[0]):
        for k in range(dim[1]):
            rga[count,:] = numpy.array(numpy.abs(([utils.RGA(Gfr)[i, k] for Gfr in freqresp])))
            count += 1  
    
    plt.figure('RGA')
    plt.clf()
    plt.gcf().set_facecolor('white')

    plot_No = 1
    
    for i in range(dim[0]):
        for k in range(dim[1]):
            plt.subplot(dim[0], dim[1], plot_No)
            plt.semilogx(w, rga[plot_No - 1])
            
            plot_No += 1
            
            plt.ylabel('|$\lambda$$_{ %s, %s}$|' % (i + 1,k + 1))
            plt.axis(axlim)
            if i == dim[0] - 1: # To avoid clutter only plot xlabel on the very bottom subplots
                plt.xlabel('Frequency [rad/unit time]')
            plt.title('Output %s vs. input %s' % (i + 1, k + 1))    

    plt.show()
    return


def rga_nm_plot(G, pairing=None, axlim=None, w_start=-2, w_end=2, points=100):
    '''
    Plots the RGA number for a specified pairing
    
    Parameters
    ----------
    G : numpy array
        plant model
    
    pairing : sparse numpy array of the same shape as G
        An array of zeros with a 1. at each required output-input pairing
        The default is a diagonal pairing with 1.'s on the diagonal
              
    Returns
    -------
    fig(RGA Number) : figure
        A figure of the RGA number for a specified pairing over
        a given frequency range
    
    Example
    -------
    # Adapted from example 3.11 pg 86 S. Skogestad
    >>> def G(s):
    ...     G = 0.01*numpy.exp(-5*s)/((s + 1.72e-4)*(4.32*s + 1))*numpy.array([[-34.54*(s + 0.0572), 1.913], [-30.22*s, -9.188*(s + 6.95e-4)]])
    ...     return G
    >>> pairing = numpy.array([[1., 0.], [0., 1.]])
    >>> rga_nm_plot(G, pairing, axlim=[None, None, 0., 1.], w_start=-5, w_end=2)

    Note
    ----
    This plotting function can only be used on square systems
    '''

    if axlim is None:
        axlim = [None, None, None, None]
    w = numpy.logspace(w_start, w_end, points)
    s = w*1j
    
    dim = numpy.shape(G(0)) # Number of rows and columns in SS transfer function
    freqresp = map(G, s)
    
    if pairing is None: # Setting a blank entry to the default of a diagonal comparison
        diag = numpy.identity(dim[0])
        print('RGA number being calculated for a diagonal pairing')
    elif not all(pairing.shape == dim):
        print('RGA_Number_Plot on plots square n by n matrices, make sure input matrix is square')
        pass
    else:
        diag = pairing

    plt.figure('RGA Number')
    plt.clf()
    plt.gcf().set_facecolor('white')        
    
    plt.semilogx(w, [numpy.sum(numpy.abs(utils.RGA(Gfr) - diag)) for Gfr in freqresp] )    
    
    plt.axis(axlim)
    plt.ylabel('||$\Lambda$(G) - I||$_{sum}$', fontsize = 15)
    plt.xlabel('Frequency (rad/unit time)')
    
    plt.show()
    return
    

def dis_rejctn_plot(G, Gd, S, axlim=None, w_start=-2, w_end=2, points=100):
    '''
    A subplot of disturbance conditition number to check for input saturation
    and a subplot of  to see if the disturbances fall withing the bounds on
    the singular values of S.
    Parameters
    ----------
    G : numpy array
        plant model
    
    Gd : numpy array
        plant disturbance model

    S : numpy array
        Sensitivity function
              
              
    Returns
    -------
    fig(Condition number and performance objective) : figure
        A figure of the disturbance condition number and the bounds imposed
        by the singular values.
    
    Note
    ----
    The disturbance condition number provides a measure of how a disturbance gd
    is aligned with the plant G. Alignment can vary between 1 and the condition
    number.
    
    For acceptable performance the singular values of S must fall below the
    inverse 2-norm of gd.
    '''

    if axlim is None:
        axlim = [None, None, None, None]

    w = numpy.logspace(w_start, w_end, points)
    s = w*1j
    
    dim = numpy.shape(G(0))    
    inv_norm_gd = numpy.zeros((dim[1],points))
    condtn_nm_gd = numpy.zeros((dim[1],points))
    for i in range(dim[1]):
        for k in range(points):
            inv_norm_gd[i,k], condtn_nm_gd[i,k] = utils.distRej(G(s[k]), Gd(s[k])[:,i])
                      
    s_min = [utils.sigmas(S(s[i]))[-1] for i in range(points)]
    s_max = [utils.sigmas(S(s[i]))[0] for i in range(points)]
    
    plt.figure('Condition number and performance objective')
    plt.clf()
    plt.gcf().set_facecolor('white')
    
    plt.subplot(2,1,1)
    for i in range(dim[1]):
        plt.loglog(w, condtn_nm_gd[i], label=('$\gamma$$_{d%s}$(G)' % (i+1)), color='blue', alpha=((i+1.)/dim[1]))
    plt.axhline(1., color='red', ls=':')  
    plt.axis(axlim)
    plt.ylabel('$\gamma$$_d$(G)')
    plt.axhline(1., color='red', ls=':')
    plt.legend()
    
    plt.subplot(2,1,2)
    for i in range(dim[1]):
        plt.loglog(w, inv_norm_gd[i], label=('1/||g$_{d%s}$||$_2$' % (i+1)), color='blue', alpha=((i+1.)/dim[1]))   
    plt.loglog(w, s_min, label=('$\sigma$$_{MIN}$'), color='green')
    plt.loglog(w, s_max, label=('$\sigma$$_{MAX}$'), color='green', alpha = 0.5)  
    plt.xlabel('Frequency (rad/unit time)')
    plt.ylabel('1/||g$_d$||$_2$')
    plt.axhline(1., color='red', ls=':')
    plt.legend()    
    
    plt.show()
    return
