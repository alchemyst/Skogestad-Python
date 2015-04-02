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
your_utilsplot_functionA(G, w_start=-5, w_end=2, axlim=[None, None, 0, 1, more_paramaters])
plt.show()

plt.figure('Example 2')     
plt.subplot(2, 1, 1)
your_utilsplot_functionB(G)
plt.subplot(2, 1, 2)
your_utilsplot_functionC(G)
plt.show()

"""

import numpy #do not abbreviate this module as np in utilsplot.py
import matplotlib.pyplot as plt
import utils
import sys


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

    if axlim is None:
        axlim = [None, None, None, None]
    plt.clf()
    plt.gcf().set_facecolor('white')

    GM, PM, wc, w_180 = utils.margins(G)

    # plotting of Bode plot and with corresponding frequencies for PM and GM
#    if ((w2 < numpy.log(w_180)) and margin):
#        w2 = numpy.log(w_180)  
    w = numpy.logspace(w_start, w_end, points)
    s = 1j*w

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

    if axlim is None:
        axlim = [None, None, None, None]
    plt.gcf().set_facecolor('white')
    
    w = numpy.logspace(w_start, w_end, points)    
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
    

def mimo_bode(Gin, w_start=-2, w_end=2, axlim=None, points=1000, Kin=None): 
    """
    Plots the max and min singular values of G and computes the crossover freq.
    
    If a controller is specified, the max and min singular values
    of S are also plotted and the bandwidth freq computed.
              
    Parameters
    ----------
    Gin : numpy matrix
        Matrix of plant transfer functions.
    
    Kin : numpy matrix
        Controller matrix (optional).
    
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
    >>> mimoBode(G, -3, 3, Kc)
    Bandwidth is a tuple of wC, wB
    (0.55557762223988783, 1.3650078065460138)
    
    """

    if axlim is None:
        axlim = [None, None, None, None]
    plt.gcf().set_facecolor('white')
    
    w = numpy.logspace(w_start, w_end, points)
    s = w*1j
    xmin = 10**w_start
    
    if Kin is None:
        plt.subplot(2, 1, 1)
    
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
    
    plt.loglog(w, Sv1, 'k-', label='Max $\sigma$(G)')
    plt.loglog(w, Sv2, 'k-', alpha=0.5, label='Min $\sigma$(G)')
    plt.axhline(1, ls=':', lw=2, color='blue')
    plt.text(xmin, 1.1, 'Mag = 1', color='blue')
    plt.axvline(wC, ls=':', lw=2, color='blue')
    plt.text(wC*1.1, ymin*1.1, 'wC', color='blue')
    plt.legend(loc='upper right', fontsize = 10, ncol=1)
    plt.xlabel('Frequency [rad/unit time]')
    plt.ylabel('Magnitude')
    plt.axis(axlim)
    plt.grid(True)
    
    if Kin is None:
        Bandwidth = wC
        print('Bandwidth = wC')
    else:
        def S(s):
            L = Kin(s)*Gin(s)
            dim = numpy.shape(Gin(0))[0]
            return numpy.linalg.inv(numpy.eye(dim) + L)  #SVD of S = 1/(I + L)

        w = numpy.logspace(w_start, w_end, points)
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
                
        plt.subplot(2, 1, 2)
        plt.loglog(w, Sv1, 'r-', label='Max $\sigma$(S)')
        plt.loglog(w, Sv2, 'r-', alpha=0.5, label='Min $\sigma$(S)')
        plt.axhline(0.707, ls=':', lw=2, color='green')
        plt.text(xmin, 0.5, 'Mag = 0.707', color='green')
        plt.axvline(wB, ls=':', lw=2, color='green')
        plt.text(wB*1.1, ymin*1.1, 'wB', color='green')
        plt.legend(loc='upper right', fontsize = 10, ncol=1)
        plt.xlabel('Frequency [rad/unit time]')
        plt.ylabel('Magnitude')
        plt.axis(axlim)
        plt.grid(True)
        Bandwidth = wC, wB
        print('Bandwidth is a tuple of wC, wB')
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

    if axlim is None:
        axlim = [None, None, None, None]
    plt.gcf().set_facecolor('white')
    
    w = numpy.logspace(w_start, w_end, points)    
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
    y_up = numpy.sqrt(1- x**2)
    y_down = -1*numpy.sqrt(1 - x**2)
    plt.plot(x, y_up, 'b:', x, y_down, 'b:', lw=2)
    plt.plot(0, 0, 'r*', ms=10)
    plt.grid(True)
    plt.axis('equal')   # Ensure the unit circle remains round on resizing the figure


def sv_plot(G, w_start=-2, w_end=2, axlim=None, points=1000, sv_all=False):
    '''
    Plot of Maximum and minimum singular values of a matirix
    
    Parameters
    ----------
    G : numpy matrix
        Plant model or sensitivity function.
        
    sv_all : boolean
        If true, plot all the singular values of the plant (optional).
              
    Returns
    -------
    Plot : matplotlib figure
    
    Note
    ----
    Can be used with the plant matrix G and the sensitivity function S
    for controlability analysis
    '''

    if axlim is None:
        axlim = [None, None, None, None]
    plt.gcf().set_facecolor('white')

    w = numpy.logspace(w_start, w_end, points)
    s = w*1j    
    
    freqresp = map(G, s)    
    sig = numpy.array([utils.sigmas(Gfr) for Gfr in freqresp])
    
    if not sv_all:
        plt.loglog(w, sig[:,0], label='$\sigma$$_{max}$')
        plt.loglog(w, sig[:,-1], label='$\sigma$$_{min}$')
    else:
        dim = numpy.shape(sig)[1]
        for sv in range(dim):
            plt.loglog(w, sig[:,sv], label='$\sigma$$_{%s}$' % sv)
    
    plt.axhline(1., color='red', ls=':')
    plt.axis(axlim)
    plt.xlabel('Frequency [rad/unit time]')
    plt.ylabel('$\sigma$')
    plt.legend()  


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

    if axlim is None:
        axlim = [None, None, None, None]
    plt.gcf().set_facecolor('white')

    w = numpy.logspace(w_start, w_end, points)
    s = w*1j    
    
    def cndtn_nm(G):
        return utils.sigmas(G)[0]/utils.sigmas(G)[-1]
    
    freqresp = map(G, s)
    
    plt.loglog(w, [cndtn_nm(Gfr) for Gfr in freqresp], label='$\gamma (G)$')
    plt.axis(axlim)
    plt.ylabel('$\gamma (G)$')
    plt.xlabel('Frequency [rad/unit time]')
    plt.axhline(10., color='red', ls=':', label='"Large" $\gamma (G) = 10$')
    plt.legend()


def rga_plot(G, w_start=-2, w_end=2, axlim=None, points=1000, plot_type='elements'):
    '''
    Plots the relative gain interaction between each output and input pairing
    
    Parameters
    ----------
    G : numpy matrix
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
    Plot : matplotlib figure
    
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
                    if axlim is not None:
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
                    if axlim is not None:
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
        
    plot_type : ['All','Element']
        Type of plot:
        
        ========       ============================
        Property       Description
        ========       ============================
        all            All the pairings on one plot
        element        Each pairing has its own plot
        =========      ============================    
              
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

    if axlim is None:
        axlim = [None, None, None, None]
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
        print("Invalid plot_type paramter.")
        sys.exit()
    

def dis_rejctn_plot(G, Gd, S=None, w_start=-2, w_end=2, axlim=None, points=1000):
    '''
    A subplot of disturbance conditition number to check for input saturation
    and a subplot of to see if the disturbances fall withing the bounds on
    the singular values of S.
    
    Parameters
    ----------
    G : numpy matrix
        Plant model.
    
    Gd : numpy matrix
        Plant disturbance model.

    S : numpy matrix
        Sensitivity function (optional, if available).
                            
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

    if axlim is None:
        axlim = [None, None, None, None]
    plt.gcf().set_facecolor('white') 

    w = numpy.logspace(w_start, w_end, points)
    s = w*1j
    
    dim = numpy.shape(Gd(0))[1]
    inv_norm_gd = numpy.zeros((dim, points))
    condtn_nm_gd = numpy.zeros((dim, points))
    for i in range(dim):
        for k in range(points):
            inv_norm_gd[i,k], condtn_nm_gd[i,k] = utils.distRej(G(s[k]), Gd(s[k])[:,i])
    
    if not S is None:
        s_min = numpy.array([utils.sigmas(S(s[i]))[-1] for i in range(points)])
        s_max = numpy.array([utils.sigmas(S(s[i]))[0] for i in range(points)])
    
    plt.subplot(2, 1, 1)
    for i in range(dim):
        plt.loglog(w, condtn_nm_gd[i], label=('$\gamma_{d%s} (G)$' % (i+1)))
    plt.axhline(1., color='red', ls=':')  
    plt.axis(axlim)
    plt.xlabel('Frequency [rad/unit time]')
    plt.ylabel('$\gamma$$_d (G)$')
    plt.axhline(1., color='red', ls=':')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for i in range(dim):
        plt.loglog(w, inv_norm_gd[i], label=('$1/||g_{d%s}||_2$' % (i+1)))
    if not S is None:
        plt.loglog(w, s_min, label='$\sigma$$_{min}$', color='green')
        plt.loglog(w, s_max, label='$\sigma$$_{max}$', color='green', alpha = 0.5)
    plt.axis(axlim) 
    plt.xlabel('Frequency [rad/unit time]')
    plt.ylabel('$1/||g_d||_2$')
    plt.legend()  
    

def input_perfect_const_plot(G, Gd, w_start=-2, w_end=2, axlim=None, points=1000, simultaneous=False):
    '''
    Plot for input constraints for perfect control.
    
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

    if axlim is None:
        axlim = [None, None, None, None]
    plt.gcf().set_facecolor('white')
    
    w = numpy.logspace(w_start, w_end, points)
    s = w*1j
    
    dim = numpy.shape(Gd(0))[1]
    perfect_control = numpy.zeros((dim, points))
    #if simultaneous: imn = np.zeros(points)
    for i in range(dim):
        for k in range(points):
            Ginv = numpy.linalg.inv(G(s[k]))
            perfect_control[i, k] = numpy.max(numpy.abs(Ginv * Gd(s[k])[:, i]))
            #if simultaneous: TODO complete induced max-norm
        plt.loglog(w, perfect_control[i], label=('$g_{d%s}$' % (i + 1)))
    
    plt.axhline(1., color='red', ls=':')      
    plt.ylabel(r'$||G^{-1}.g_d||_{max}$')
    plt.xlabel('Frequency [rad/unit time]')         
    plt.grid(True)
    plt.axis(axlim)
    plt.legend()


def input_acceptable_const_plot(G, Gd, w_start=-2, w_end=2, axlim=None, points=1000):
    '''
    Subbplots for input constraints for accepatable control.
    
    Parameters
    ----------
    G : numpy matrix
        Plant model.
    
    Gd : numpy matrix
        Plant disturbance model.

    Returns
    -------
    
    Plot : matplotlib figure
    '''

    if axlim is None:
        axlim = [None, None, None, None]
    plt.gcf().set_facecolor('white')
    
    w = numpy.logspace(w_start, w_end, points)
    s = w*1j    
    
    freqresp = map(G, s) 
    sig = numpy.array([utils.sigmas(Gfr) for Gfr in freqresp])  
    
    plot_No = 1
    
    dimGd = numpy.shape(Gd(0))[1]
    dimG = numpy.shape(G(0))[0]
    acceptable_control = numpy.zeros((dimGd, dimG, points))
    for j in range(dimGd):
        for i in range(dimG):
            for k in range(points):
                U, _, _ = utils.SVD(G(s[k]))
                acceptable_control[j, i, k] = numpy.abs(U[:, i].H * Gd(s[k])[:, j]) - 1
            plt.subplot(dimGd, dimG, plot_No)
            plt.plot(w, acceptable_control[j, i], label=('$|u_%s^H.g_{d%s}|-1$' % (i + 1, j + 1)))
            plt.loglog(w, sig[:, i], label=('$\sigma_%s$' % (i + 1)))
            plt.xlabel('Frequency [rad/unit time]')
            plt.grid(True)
            plt.axis(axlim)
            plt.legend()
            plot_No += 1


def step(G, t_end=100, initial_val=0, points=1000):
    '''
    This function is similar to the MatLab step function. Models must be
    defined as tf objects
    
    Parameters
    ----------
    G : tf
        Plant transfer function.
        
    t_end : integer
        Time period which the step response should occur (optional).
    
    initial_val : integer
        Starting value to evalaute step response (optional).
          
    Returns
    -------
    
    Plot : matplotlib figure
    '''
    plt.gcf().set_facecolor('white')
    
    rows = numpy.shape(G(0))[0]
    columns = numpy.shape(G(0))[1]

    s = utils.tf([1, 0])
    system = G(s)
    
    fig = plt.figure(1, figsize=(12, 8))
    bigax = fig.add_subplot(111)
    bigax.spines['top'].set_color('none')
    bigax.spines['bottom'].set_color('none')
    bigax.spines['left'].set_color('none')
    bigax.spines['right'].set_color('none')
    bigax.tick_params(labelcolor='grey', top='off', bottom='off',
                      left='off', right='off')
    plt.setp(bigax.get_xticklabels(), visible=False)
    plt.setp(bigax.get_yticklabels(), visible=False)

    cnt = 0
    tspace = numpy.linspace(0, t_end, points)
    for i in range(rows):
        for k in range(columns):
            cnt += 1
            nulspace = numpy.zeros(points)
            ax = fig.add_subplot(rows + 1, columns, cnt)
            tf = system[i, k]
            if all(tf.numerator) != 0:
                realstep = numpy.real(tf.step(initial_val, tspace))
                ax.plot(realstep[0], realstep[1])
            else:
                ax.plot(tspace, nulspace)
            
            if i == 0:
                xax = ax
            else:
                ax.sharex = xax
                
            if k == 0:
                yax = ax
            else:
                ax.sharey =yax
            
            if i == range(rows)[-1]:            
                plt.setp(ax.get_xticklabels(), visible=True, fontsize=10)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
                
            plt.setp(ax.get_yticklabels(), fontsize=10)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 * 1.05 - 0.05,
                             box.width * 0.85, box.height])
#            if k == 0:            
#                plt.setp(ax.get_yticklabels(), visible=True)
#            else:
#                plt.setp(ax.get_yticklabels(), visible=False)
    box = bigax.get_position()    
    bigax.set_position([box.x0 - 0.05, box.y0 - 0.02,
                        box.width * 1.1, box.height * 1.1])        
    bigax.set_ylabel('Output magnitude')
    bigax.set_xlabel('Time')   
    
    
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
        Property    Description
        ========    ==================================
        S           Sensitivity function
        T           Complementary sensitivity function
        L           Loop function
        ========    ==================================
              
    Returns
    -------
    Plot : matplotlib figure
    
    '''

    if axlim is None:
        axlim = [None, None, None, None]
    plt.gcf().set_facecolor('white') 
    
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
    plt.xlabel('Frequency [rad/unit time]')
    plt.legend(["Kc=0.2", "Kc=0.5", "Kc=0.8"],loc=4)
               
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

    if axlim is None:
        axlim = [None, None, None, None]
    plt.gcf().set_facecolor('white')       
    
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
    ...     return numpy.linalg.inv((numpy.eye(2) + L(s)))      #SVD of S =
    #  1/(I + L)
    ... 
    >>> #utils.perf_Wp(S, 0.05, 0.2, -3, 1)
    
    '''

    if axlim is None:
        axlim = [None, None, None, None]
    plt.gcf().set_facecolor('white') 
    
    w = numpy.logspace(w_start, w_end, points)
    s = w*1j
    
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
