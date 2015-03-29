# -*- coding: utf-8 -*-
"""
Inputs
------
The default inputs to a plotting function in this script include:

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
         
show : boolean
       If false, the plot is not automatically shown and can be edit externally
       for special cases.

Plots
-----
bode: Shows the bode plot for a plant model

bodeclosedloop: Shows the bode plot for a controller model

mimo_bode: Plots the max and min singular values of G and computes the crossover freq

nyquist_plot: TODO

mino_nyquist_plot: Nyquist stability plot for MIMO system
    
sv_plot : Maximum and minimum singular values of a matirix
    
condtn_nm_plot : A plot of the condition number for a specified diagonal 
    
rga_plot: A plot of the relative gain interactions for a matrix over a given frequency

rga_nm_plot: A plot of the RGA number for a given pairing
    
dis_rejctn_plot : A plot of the disturbance condition number and the bounds imposed
    by the singular values.
    
freq_step_response_plot: A subplot for both the frequnecy response and step
    response for a controlled plant
    
step_response_plot: A plot of the step response of a transfer function
   
perf_Wp_plot: MIMO sensitivity S and performance weight Wp plotting funtion
   
"""

import sys

import numpy #do not abbreviate this module as np in utilsplot.py
import matplotlib.pyplot as plt

import utils


def bode(G, w_start=-2, w_end=2, axlim=None, points=100, margin=False):
    """ 
    Shows the bode plot for a plant model
    
    Parameters
    ----------
    G : tf
        plant transfer function
    margin : boolean
        show the cross over frequencies on the plot (optional)        
          
    Returns
    -------
    GM : array containing a real number      
        gain margin
    PM : array containing a real number           
        phase margin         
    """

    GM, PM, wc, w_180 = utils.margins(G)

    # plotting of Bode plot and with corresponding frequencies for PM and GM
#    if ((w2 < numpy.log(w_180)) and margin):
#        w2 = numpy.log(w_180)  
    w = numpy.logspace(w_start, w_end, points)
    s = 1j*w

    plt.figure('Bode Plot')
    # Magnitude of G(jw)
    plt.subplot(211)
    gains = numpy.abs(G(s))
    plt.loglog(w, gains)
    if margin:
        plt.axvline(w_180, color='black')
        plt.text(w_180, numpy.average([numpy.max(gains), numpy.min(gains)]), r'$\angle$G(jw) = -180$\degree$')
    plt.axhline(1., color='red')
    plt.grid()
    plt.ylabel('Magnitude')

    # Phase of G(jw)
    plt.subplot(212)
    phaseangle = utils.phase(G(s), deg=True)
    plt.semilogx(w, phaseangle)
    if margin:
        plt.axvline(wc, color='black')
        plt.text(wc, numpy.average([numpy.max(phaseangle), numpy.min(phaseangle)]), '|G(jw)| = 1')
    plt.axhline(-180., color='red')
    plt.grid()
    plt.ylabel('Phase')
    plt.xlabel('Frequency [rad/unit time]')
    

    return GM, PM
    
def bodeclosedloop(G, K, w1, w2, label='Figure', margin=False):
    """ 
    Shows the bode plot for a controller model
    
    Parameters
    ----------
    G : tf
        plant transfer function
    K : tf
        controller transfer function
    w1 : real
        start frequency
    w2 : real
        end frequency
    label : string
        title for the figure (optional)
    margin : boolean
        show the cross over frequencies on the plot (optional)             
    """
    
    w = numpy.logspace(w1, w2, 1000)    
    L = G(1j*w) * K(1j*w)
    S = utils.feedback(1, L)
    T = utils.feedback(L, 1)
    
    plt.figure(label)
    plt.subplot(2, 1, 1)
    plt.loglog(w, abs(L))
    plt.loglog(w, abs(S))
    plt.loglog(w, abs(T))
    plt.grid()
    plt.ylabel("Magnitude")
    plt.legend(["L", "S", "T"],
               bbox_to_anchor=(0, 1.01, 1, 0), loc=3, ncol=3)
    
    if margin:        
        plt.plot(w, 1/numpy.sqrt(2) * numpy.ones(len(w)), linestyle='dotted')
        
    plt.subplot(2, 1, 2)
    plt.semilogx(w, utils.phase(L, deg=True))
    plt.semilogx(w, utils.phase(S, deg=True))
    plt.semilogx(w, utils.phase(T, deg=True))
    plt.grid()
    plt.ylabel("Phase")
    plt.xlabel("Frequency [rad/s]")  
    

def mimoBode(Gin, wStart, wEnd, Kin=None): 
    """
    Plots the max and min singular values of G and computes the crossover freq.
    
    If a controller is specified, the max and min singular values
    of S are also plotted and the bandwidth freq computed.
              
    Parameters
    ----------
    Gin : numpy array
        Matrix of plant transfer functions.
    
    wStart : float
        Minimum power of w for the frequency range in rad/time. 
        eg: for w startig at 10e-3, wStart = -3.
        
    wEnd : float
        Maximum value of w for the frequency range in rad/time. 
        eg: for w ending at 10e3, wStart = 3.
    
    Kin : numpy array
        Controller matrix (optional).
    
    Returns
    -------
    wC : real
        Crossover frequency.
        
    wB : real
        Bandwidth frequency.
        
    Plot : matplotlib plot
        Bode plot of singular values of G and S(optional).
    
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
    >>> mimoBode(G, -3, 3, Kc)
    Bandwidth is a tuple of wC, wB
    (0.55557762223988783, 1.3650078065460138)
    
    """
    xmin = 10**wStart
    xmax = 10**wEnd
    w = numpy.logspace(wStart, wEnd, 1000)
    s = w*1j
    Sv1 = numpy.zeros(len(w), dtype=complex)
    Sv2 = numpy.zeros(len(w), dtype=complex)
    f = 0
    wC = 0
    for i in range(len(w)):
        Sv1[i] = utils.sigmas(Gin(s[i]))[0]
        Sv2[i] = utils.sigmas(Gin(s[i]))[-1]
        if f < 1 and Sv2[i] < 1:
            wC = w[i]
            f = 1
    ymin = numpy.min(Sv2)
    plt.figure('MIMO Bode')
    plt.clf()
    plt.loglog(w, Sv1, 'k-', label='Max $\sigma$(G)')
    plt.loglog(w, Sv2, 'k-', alpha=0.5, label='Min $\sigma$(G)')
    plt.axhline(1, ls=':', lw=2, color='blue')
    plt.text(xmin, 1.1, 'Mag = 1', color='blue')
    plt.axvline(wC, ls=':', lw=2, color='blue')
    plt.text(wC*1.1, ymin*1.1, 'wC', color='blue')
    plt.legend(loc='upper right', fontsize = 10, ncol=1)
    plt.xlabel('Frequency [rad/time]')
    plt.ylabel('Magnitude')
    plt.axis([xmin, xmax, None, None])
    plt.grid(True)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    
    if Kin is None:
        Bandwidth = wC
        print('Bandwidth = wC')
    else:
        def S(s):
            L = Kin(s)*Gin(s)
            dim1, dim2 = numpy.shape(Gin(0))
            return numpy.linalg.inv(numpy.eye(dim1) + L)  #SVD of S = 1/(I + L)
        w = numpy.logspace(wStart, wEnd, 1000)
        s = w*1j
        Sv1 = numpy.zeros(len(w), dtype=complex)
        Sv2 = numpy.zeros(len(w), dtype=complex)
        f = 0
        wB = 0
        for i in range(len(w)):
            Sv1[i] = utils.sigmas(S(s[i]))[0]
            Sv2[i] = utils.sigmas(S(s[i]))[-1]
            if f < 1 and Sv1[i] > 0.707:
                wB = w[i]
                f = 1
        plt.figure()
        plt.loglog(w, Sv1, 'r-', label='Max $\sigma$(S)')
        plt.loglog(w, Sv2, 'r-', alpha=0.5, label='Min $\sigma$(S)')
        plt.axhline(0.707, ls=':', lw=2, color='green')
        plt.text(xmin, 0.5, 'Mag = 0.707', color='green')
        plt.axvline(wB, ls=':', lw=2, color='green')
        plt.text(wB*1.1, ymin*1.1, 'wB', color='green')
        plt.legend(loc='upper right', fontsize = 10, ncol=1)
        plt.xlabel('Frequency [rad/time]')
        plt.ylabel('Magnitude')
        plt.axis([xmin, xmax, None, None])
        plt.grid(True)
        fig = plt.gcf()
        BG = fig.patch
        BG.set_facecolor('white')
        Bandwidth = wC, wB
        print('Bandwidth is a tuple of wC, wB')
    return Bandwidth


def mino_nyquist_plot(L, axLim, wStart, wEnd):
    """
    Nyquist stability plot for MIMO system.
    
    Parameters
    ----------
    L : numpy array
        Closed loop transfer function matrix as a function of s, i.e. def L(s).
    
    axLim : float
        Axis limit for square axis.  axis will run from -axLim to +axLim.
    
    wStart : float
        Minimum power of w for the frequency range in rad/time. 
        eg: for w startig at 10e-3, wStart = -3.
        
    wEnd : float
        Maximum value of w for the frequency range in rad/time. 
        eg: for w ending at 10e3, wStart = 3.
        
    Returns
    -------
    Nyquist stability plot.
    
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
    w = numpy.logspace(wStart, wEnd, 1000)    
    Lin = numpy.zeros((len(w)), dtype=complex)
    x = numpy.zeros((len(w)))
    y = numpy.zeros((len(w)))
    dim = numpy.shape(L(0.1))
    for i in range(len(w)):        
        Lin[i] = numpy.linalg.det(numpy.eye(dim[0]) + L(w[i]*1j))
        x[i] = numpy.real(Lin[i])
        y[i] = numpy.imag(Lin[i])        
    plt.figure('MIMO Nyquist Plot')
    plt.clf()
    plt.plot(x, y, 'k-', lw=1)
    plt.xlabel('Re G(wj)')
    plt.ylabel('Im G(wj)')
    # plotting a unit circle
    x = numpy.linspace(-1, 1, 200)
    y_up = numpy.sqrt(1- x**2)
    y_down = -1*numpy.sqrt(1 - x**2)
    plt.plot(x, y_up, 'b:', x, y_down, 'b:', lw=2)
    plt.plot(0, 0, 'r*', ms=10)
    plt.grid(True)
    n = axLim           # Sets x-axis limits
    plt.axis('equal')   # Ensure the unit circle remains round on resizing the figure
    plt.axis([-n, n, -n, n])
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')


def sv_plot(G, w_start=-2, w_end=2, axlim=None, points=100):
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
    
    plt.semilogx(w, [utils.sigmas(Gfr)[0] for Gfr in freqresp], label='$\sigma$$_{MAX}$', color='blue')
    plt.semilogx(w, [utils.sigmas(Gfr)[-1] for Gfr in freqresp], label='$\sigma$$_{MIN}$', color='blue', alpha=0.5)
    plt.xlabel('Frequency (rad/unit time)')
    
    plt.axhline(1., color='red', ls=':')
    plt.legend()  
    plt.show()
    return


def condtn_nm_plot(G, w_start=-2, w_end=2, axlim=None, points=100):
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
        return utils.sigmas(G)[0]/utils.sigmas(G)[-1]
    
    freqresp = map(G, s)
    
    plt.figure('Condition number')
    plt.clf()
    plt.gcf().set_facecolor('white')
    
    plt.semilogx(w, [cndtn_nm(Gfr) for Gfr in freqresp], label='$\sigma$$_{MAX}$/$\sigma$$_{MIN}$')
    plt.axis(axlim)
    plt.ylabel('$\gamma$(G)', fontsize = 15)
    plt.xlabel('Frequency (rad/unit time)')
    plt.axhline(10., color='red', ls=':', label='$\gamma$(G) > 10')
    plt.legend()
    plt.show()
    return


def rga_plot(G, w_start=-2, w_end=2, axlim=None, points=100, show=True, plot_type='elements'):
    '''
    Plots the relative gain interaction between each output and input pairing
    
    Parameters
    ----------
    G : numpy array
        Plant model.
        
    plot_type : ['All','Output','Input','Element']
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
    fig(RGA) : figure
        A figure of subplots for each interaction between an output and
        an input.
    
    Example
    -------
    # Adapted from example 3.11 pg 86 S. Skogestad
    >>> def G(s):
    ...     G = 0.01**(-5*s)/((s + 1.72e-4)*(4.32*s + 1))*numpy.matrix([[-34.54*(s + 0.0572), 1.913], [-30.22*s, -9.188*(s + 6.95e-4)]])
    ...     return G
    >>> rga_plot(G, w_start=-5, w_end=2, axlim=[None, None, 0., 1.])
    
    
    Note
    ----
    This entry draws on and improves RGA.py.
    If accepted, then RGA.py could be removed from the repository
    '''

    if axlim is None:
        axlim = [None, None, None, None]
    
    if show:
        plt.figure('RGA')
        plt.clf()
        plt.gcf().set_facecolor('white') 

    w = numpy.logspace(w_start, w_end, points)
    s = w*1j
    
    dim = numpy.shape(G(0)) # Number of rows and columns in SS transfer function    
    freqresp = map(G, s)
    
    plot_No = 1
        
    if plot_type == 'elements':
        for i in range(dim[0]):
            for j in range(dim[1]):
                plt.subplot(dim[0], dim[1], plot_No)
                plt.title('Output %s vs. input %s' % (i + 1, j + 1))  
                plt.semilogx(w, numpy.array(numpy.abs(([utils.RGA(Gfr)[i, j] for Gfr in freqresp]))))
                
                plot_No += 1
                
                plt.axis(axlim)
                plt.ylabel('|$\lambda$$_{ %s, %s}$|' % (i + 1,j + 1))
                if i == dim[0] - 1: # To avoid clutter only plot xlabel on the very bottom subplots
                    plt.xlabel('Frequency [rad/unit time]')
                        
    elif plot_type == 'outputs': #i
        for i in range(dim[0]):
            plt.subplot(dim[1], 1, plot_No)
            plt.title('Output %s vs. input j' % (i + 1))
            rgamax = []
            for j in range(dim[1]):
                rgas = numpy.array(numpy.abs(([utils.RGA(Gfr)[i, j] for Gfr in freqresp])))
                plt.semilogx(w, rgas, label='$\lambda$$_{%s, %s}$' % (i + 1, j + 1))
                rgamax.append(max(rgas))

                if j == dim[1] - 1: #self-scaling algorithm
                    if axlim != None:
                        plt.axis(axlim)
                    else:
                        plt.axis([None, None, None, max(rgamax)])
                
            plt.ylabel('|$\lambda$$_{%s, j}$|' % (i + 1))
            plt.legend()
            if i == dim[0] - 1: # To avoid clutter only plot xlabel on the very bottom subplots
                plt.xlabel('Frequency [rad/unit time]')  
            plot_No += 1     
            
    elif plot_type == 'inputs': #j
        for j in range(dim[1]):
            plt.subplot(dim[0], 1, plot_No)
            plt.title('Output i vs. input %s' % (j + 1))
            rgamax = []
            for i in range(dim[0]):
                rgas = numpy.array(numpy.abs(([utils.RGA(Gfr)[i, j] for Gfr in freqresp])))
                plt.semilogx(w, rgas, label='$\lambda$$_{%s, %s}$' % (i + 1, j + 1))
                rgamax.append(max(rgas))

                if i == dim[1] - 1: #self-scaling algorithm
                    if axlim != None:
                        plt.axis(axlim)
                    else:
                        plt.axis([None, None, None, max(rgamax)])
                
            plt.ylabel('|$\lambda$$_{i, %s}$|' % (j + 1))
            plt.legend()
            if j == dim[1] - 1: # To avoid clutter only plot xlabel on the very bottom subplots
                plt.xlabel('Frequency [rad/unit time]')   
            plot_No += 1  
            
    elif plot_type == 'all':
        for i in range(dim[0]):
            for j in range(dim[1]):
                plt.semilogx(w, numpy.array(numpy.abs(([utils.RGA(Gfr)[i, j] for Gfr in freqresp]))))
                plt.axis(axlim)            
                plt.ylabel('|$\lambda$$_{i,j}$|')
                plt.xlabel('Frequency [rad/unit time]')
                
    else:
        print("Invalid plot_type paramter.")
        sys.exit()        

    if show: plt.show()


def rga_nm_plot(G, pairing_list=None, pairing_names=None, w_start=-2, w_end=2, axlim=None, points=100, show=True, plot_type='all'):
    '''
    Plots the RGA number for a specified pairing
    
    Parameters
    ----------
    G : numpy matrix
        Plant model
    
    pairing_list : list of sparse numpy matrixes of the same shape as G
        An array of zeros with a 1. at each required output-input pairing
        The default is a diagonal pairing with 1.'s on the diagonal
        
    plot_type : ['All','Element']
        Type of plot.
        
        =========      ============================
        plot_type      Type of plot
        =========      ============================
        all            All the pairings on one plot
        element        Each pairing has its own plot
        =========      ============================    
              
    Returns
    -------
    fig(RGA Number) : figure
        A figure of the RGA number for a specified pairing over
        a given frequency range
    
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

    if axlim is None:
        axlim = [None, None, None, None]
    
    if show:
        plt.figure('RGA Number')
        plt.clf()
        plt.gcf().set_facecolor('white')  
        
    w = numpy.logspace(w_start, w_end, points)
    s = w*1j
    
    dim = numpy.shape(G(0)) # Number of rows and columns in SS transfer function
    freqresp = map(G, s)
    
    if pairing_list is None: # Setting a blank entry to the default of a diagonal comparison
        pairing_list = numpy.identity(dim[0])
        pairing_names ='Diagonal pairing'
    else:
        for pairing in pairing_list:
            if pairing.shape != dim:
                print('RGA_Number_Plot on plots square n by n matrices, make sure input matrix is square')
                sys.exit()
         
    plot_No = 0
    
    if plot_type == 'all':
        for p in pairing_list:    
            plt.semilogx(w, [utils.RGAnumber(Gfr, p) for Gfr in freqresp], label=pairing_names[plot_No])  
            plot_No += 1     
        plt.axis(axlim)
        plt.ylabel('||$\Lambda$(G) - I||$_{sum}$')
        plt.xlabel('Frequency (rad/unit time)')
        plt.legend() 

    elif plot_type == 'element':
        pcount = numpy.shape(pairing_list)[0] # pairing_list.count not accessible
        for p in pairing_list:   
            plot_No += 1   
            plt.subplot(1, pcount, plot_No)            
            plt.semilogx(w, [utils.RGAnumber(Gfr, p) for Gfr in freqresp])
            plt.axis(axlim)
            plt.title(pairing_names[plot_No - 1])            
            plt.xlabel('Frequency (rad/unit time)')
            if plot_No == 1:
                plt.ylabel('||$\Lambda$(G) - I||$_{sum}$')
    else:
        print("Invalid plot_type paramter.")
        sys.exit()    
        
    if show: plt.show()
    

def dis_rejctn_plot(G, Gd, S, w_start=-2, w_end=2, axlim=None, points=100):
    '''
    A subplot of disturbance conditition number to check for input saturation
    and a subplot of to see if the disturbances fall withing the bounds on
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
    plt.loglog(w, s_min, label='$\sigma$$_{MIN}$', color='green')
    plt.loglog(w, s_max, label='$\sigma$$_{MAX}$', color='green', alpha = 0.5)
    plt.xlabel('Frequency (rad/unit time)')
    plt.ylabel('1/||g$_d$||$_2$')
    plt.axhline(1., color='red', ls=':')
    plt.legend()    
    
    plt.show()
    return


def freq_step_response_plot(G, K, Kc, t_end=50, t_points=100, freqtype='S', heading='', w_start=-2, w_end=2, axlim=None, points=1000):
    '''
    A subplot function for both the frequnecy response and step response for a
    controlled plant
    
    Parameters
    ----------
    G : tf
        plant transfer function
    
    K : tf
        controller transfer function
        
    Kc : integer
        controller constant 

    freqtype : string (optional)
        S = Sensitivity function; T = Complementary sensitivity function; L =
        Loop function
              
    Returns
    -------
    fig(Frequency and step response) : figure
        A subplot for both the frequnecy response and step response for a
        controlled plant
    
    '''
    
    if axlim is None:
        axlim = [None, None, None, None]
    
    plt.figure(heading)
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
    
    w = numpy.logspace(w_start, w_end, points)
    wi = w * 1j
    i = 0
    for F in Fs:
        plt.loglog(w, abs(F(wi)), label='Kc={d%s}=' % Kc[i])
        i =+ 1
    plt.axis(axlim)
    plt.grid(b=None, which='both', axis='both')
    plt.xlabel('Frequency [rad/s]')
    plt.legend(["Kc=0.2", "Kc=0.5", "Kc=0.8"],loc=4)
               
    plt.subplot(1, 2, 2)
    plt.title('(b) Response to step in reference')
    tspan = numpy.linspace(0, t_end, t_points)
    for T in Ts:
        [t, y] = T.step(0, tspan)
        plt.plot(t, y)
    plt.plot(tspan, 0 * numpy.ones(t_points), ls='--')
    plt.plot(tspan, 1 * numpy.ones(t_points), ls='--')
    plt.axis(axlim)
    plt.xlabel('Time [sec]')
    plt.ylabel('$y(t)$')
    
    plt.show()


def step_response_plot(Y, U, t_end=50, initial_val=0, timedim='sec', axlim=None, points=1000, constraint=None, method='numeric'):
    '''
    A plot of the step response of a transfer function
    
    Parameters
    ----------
    Y : tf
        output transfer function
        
    U : tf
        input transfer function
    
    initial_val : integer
        starting value to evalaute step response (optional)
        
    constraint : float
        the upper limit the step response cannot exceed. is only calculated
        if a value is specified (optional)
        
    method : ['numeric','analytic']
        the method that is used to calculate a constrainted response. a
        constraint value is required (optional)

    Returns
    -------
    fig(Step response) : figure
    
    '''    
    if axlim is None:
        axlim = [None, None, None, None]       
    
    [t,y] = utils.tf_step(Y, t_end, initial_val)
    plt.plot(t,y)
    
    [t,y] = utils.tf_step(U, t_end, initial_val)
    plt.plot(t,y)
    
    if constraint == None:
        plt.legend(['$y(t)$','$u(t)$'])  
    else:
        [t,y] = utils.tf_step(U, t_end, initial_val, points, constraint, Y, method)
        plt.plot(t,y[0])
        plt.plot(t,y[1])
        
        plt.legend(['$y(t)$','$u(t)$','$u(t) const$','$y(t) const$']) #con = constraint
        
    plt.plot([0, t_end], numpy.ones(2),'--')    
    
    plt.axis(axlim)
    plt.xlabel('Time [' + timedim + ']')  


def perf_Wp_plot(S, wB_req, maxSSerror, wStart, wEnd):
    """
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
        A plot of the sensitivity function and the performance weight across the
        frequency range specified.
        
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
    ...     return numpy.linalg.inv((numpy.eye(2) + L(s)))      #SVD of S =
    #  1/(I + L)
    ... 
    >>> #utils.perf_Wp(S, 0.05, 0.2, -3, 1)
    
    """
    w = numpy.logspace(wStart, wEnd, 1000)
    s = w*1j
    magPlotS1 = numpy.zeros((len(w)))
    magPlotS3 = numpy.zeros((len(w)))
    Wpi = numpy.zeros((len(w)))
    f = 0                                    #f for flag
    for i in range(len(w)):
        U, Sv, V = utils.SVD(S(s[i]))
        magPlotS1[i] = Sv[0]
        magPlotS3[i] = Sv[-1]
        if f < 1 and magPlotS1[i] > 0.707:
            wB = w[i]
            f = 1
    for i in range(len(w)):
        Wpi[i] = utils.Wp(wB_req, maxSSerror, s[i])                                      
    plt.figure('MIMO sensitivity S and performance weight Wp')
    plt.clf()
    plt.subplot(211)
    plt.loglog(w, magPlotS1, 'r-', label='Max $\sigma$(S)')
    plt.loglog(w, 1./Wpi, 'k:', label='|1/W$_P$|', lw=2.)
    plt.axhline(0.707, color='green', ls=':', lw=2, label='|S| = 0.707')
    plt.axvline(wB_req, color='blue', ls=':', lw=2)
    plt.text(wB_req*1.1, 7, 'req wB', color='blue', fontsize=10)
    plt.axvline(wB, color='green')
    plt.text(wB*1.1, 0.12, 'wB = %s rad/s'%(numpy.round(wB,3)), color='green', fontsize=10)
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Magnitude')
    plt.axis([None, None, 0.1, 10])
    plt.legend(loc='upper left', fontsize=10, ncol=1)
    plt.grid(True)
    plt.subplot(212)
    plt.semilogx(w, magPlotS1*Wpi, 'r-', label='|W$_P$S|')
    plt.axhline(1, color='blue', ls=':', lw=2)
    plt.axvline(wB_req, color='blue', ls=':', lw=2, label='|W$_P$S| = 1')
    plt.text(wB_req*1.1, numpy.max(magPlotS1*Wpi)*0.95, 'req wB', color='blue', fontsize=10)
    plt.axvline(wB, color='green')
    plt.text(wB*1.1, 0.12, 'wB = %s rad/s'%(numpy.round(wB,3)), color='green', fontsize=10)
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Magnitude')
    plt.legend(loc='upper right', fontsize=10, ncol=1)
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    plt.grid(True)
    return wB
              