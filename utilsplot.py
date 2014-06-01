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

"""

import numpy
import utils
import matplotlib.pyplot as plt
           
                
def G(s):   # Example 3.11 transfer function to test RGA_Plot
    '''The transfer function model of the system, as a function of s'''
    G = 0.01*numpy.exp(-5*s)/((s + 1.72e-4)*(4.32*s + 1))*numpy.array([[-34.54*(s + 0.0572), 1.913], [-30.22*s, -9.188*(s + 6.95e-4)]])
    return G


def RGA_Plot(G, axlim=[None, None, None, None], w_start=-2, w_end=2, points=100):
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
    >>> RGA_Plot(G, axlim=[None, None, 0., 1.], w_start=-5, w_end=2)
    
    
    Note
    ----
    This entry draws on and improves RGA.py.
    If accepted, then RGA.py could be removed from the repository
    '''
    w = numpy.logspace(w_start, w_end, points)
    s = w*1j
    
    dim = numpy.shape(G(0)) # Number of rows and columns in SS transfer function    
    freqresp = map(G, s)
    
    count = 0 # Arrange the individual RGA responses to be compatible with the plotting order of plt.subplot
    rga = numpy.zeros([dim[0]*dim[1],points])
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
            plt.subplot(dim[0],dim[1],plot_No)
            plt.semilogx(w, rga[plot_No-1])
            
            plot_No += 1
            
            plt.ylabel('|$\lambda$$_{ %s, %s}$|' % (i+1,k+1))
            plt.axis(axlim)
            if i == dim[0] - 1: # To avoid clutter only plot xlabel on the very bottom subplots
                plt.xlabel('Frequency [rad/unit time]')
            plt.title('Output %s vs. input %s' % (i+1, k+1))    

    plt.show()
    return
    
