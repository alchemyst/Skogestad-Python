# -*- coding: utf-8 -*-
'''
Created on Jan 27, 2012

@author: Carl Sandrock
'''

import numpy #do not abbreviate this module as np in utils.py
import matplotlib.pyplot as plt
from scipy import optimize, signal

def circle(cx, cy, r):
    npoints = 100
    theta = numpy.linspace(0, 2*numpy.pi, npoints)
    y = cy + numpy.sin(theta)*r
    x = cx + numpy.cos(theta)*r
    return x, y


def arrayfun(f, A):
    """
    Recurses down to scalar elements in A, then applies f, returning lists
    containing the result.
    """
    if len(A.shape) == 0:
        return f(A)
    else:
        return [arrayfun(f, b) for b in A]


def listify(A):
    return [A]


def gaintf(K):
    r = tf(arrayfun(listify, K), arrayfun(listify, numpy.ones_like(K)))
    return r


def findst(G, K):
    """ 
    Find S and T given a value for G and K.
    
    Parameters
    ----------
    G : numpy array
        Matrix of transfer functions.
    K : numpy array
        Matrix of controller functions.
    
    Returns
    -------
    S : numpy array
        Matrix of sensitivities.
    T : numpy array
        Matrix of complementary sensitivities.
    
   """
    L = G*K
    I = numpy.eye(G.outputs, G.inputs)
    S = numpy.linalg.inv(I + L)
    T = S*L
    return S, T


def phase(G, deg=False):
    return numpy.unwrap(numpy.angle(G, deg=deg), 
                        discont=180 if deg else numpy.pi)


def Closed_loop(Kz, Kp, Gz, Gp):
    """
    Kz & Gz is the polynomial constants in the numerator
    Kp & Gp is the polynomial constants in the denominator
    """

    # calculating the product of the two polynomials in the numerator
    # and denominator of transfer function GK
    Z_GK = numpy.polymul(Kz, Gz)
    P_GK = numpy.polymul(Kp, Gp)

    # calculating the polynomial of closed loop
    # sensitivity function s = 1/(1+GK)
    Zeros_poly = Z_GK
    Poles_poly = numpy.polyadd(Z_GK, P_GK)
    return Zeros_poly, Poles_poly


def RGA(Gin):
    """ 
    Computes the Relative Gain Array of a matrix.
    
    
    Parameters
    ----------
    Gin : numpy array
        Transfer function matrix.
        
    Returns
    -------
    RGA matrix : matrix
        RGA matrix of complex numbers.
    
    Example
    -------
    >>> G = numpy.array([[1, 2],[3, 4]])
    >>> RGA(G)
    array([[-2.,  3.],
           [ 3., -2.]])

    """
    G = numpy.asarray(Gin)
    Ginv = numpy.linalg.pinv(G)
    return G*Ginv.T


def plot_freq_subplot(plt, w, direction, name, color, figure_num):
    plt.figure(figure_num)
    N = direction.shape[0]
    for i in range(N):
        #label = '%s Input Dir %i' % (name, i+1)

        plt.subplot(N, 1, i + 1)
        plt.title(name)
        plt.semilogx(w, direction[i, :], color)


def polygcd(a, b):
    """
    Find the Greatest Common Divisor of two polynomials
    using Euclid's algorithm:
    http://en.wikipedia.org/wiki/Polynomial_greatest_common_divisor#Euclidean_algorithm

    >>> a = numpy.poly1d([1, 1]) * numpy.poly1d([1, 2])
    >>> b = numpy.poly1d([1, 1]) * numpy.poly1d([1, 3])
    >>> polygcd(a, b)
    poly1d([ 1.,  1.])

    >>> polygcd(numpy.poly1d([1, 1]), numpy.poly1d([1]))
    poly1d([ 1.])
    """
    if len(a) > len(b):
        a, b = b, a
    while len(b) > 0 or abs(b[0]) > 0:
        q, r = a/b
        a = b
        b = r
    return a/a[len(a)]


class tf(object):
    """
    Very basic transfer function object

    Construct with a numerator and denominator:

    >>> G = tf(1, [1, 1])
    >>> G
    tf([ 1.], [ 1.  1.])

    >>> G2 = tf(1, [2, 1])

    The object knows how to do:

    addition

    >>> G + G2
    tf([ 3.  2.], [ 2.  3.  1.])
    >>> G + G # check for simplification
    tf([ 2.], [ 1.  1.])

    multiplication

    >>> G * G2
    tf([ 1.], [ 2.  3.  1.])

    division

    >>> G / G2
    tf([ 2.  1.], [ 1.  1.])

    Deadtime is supported:

    >>> G3 = tf(1, [1, 1], deadtime=2)
    >>> G3
    tf([ 1.], [ 1.  1.], deadtime=2)

    Note we can't add transfer functions with different deadtime:

    >>> G2 + G3
    Traceback (most recent call last):
        ...
    ValueError: Transfer functions can only be added if their deadtimes are the same

    It is sometimes useful to define

    >>> s = tf([1, 0])
    >>> 1 + s
    tf([ 1.  1.], [ 1.])

    >>> 1/(s + 1)
    tf([ 1.], [ 1.  1.])
    """

    def __init__(self, numerator, denominator=1, deadtime=0, name='', u='', y=''):
        """
        Initialize the transfer function from a
        numerator and denominator polynomial
        """
        # TODO: poly1d should be replaced by np.polynomial.Polynomial
        self.numerator = numpy.poly1d(numerator)
        self.denominator = numpy.poly1d(denominator)
        self.simplify()
        self.deadtime = deadtime
        self.name = name
        self.u = u
        self.y = y

    def inverse(self):
        """
        Inverse of the transfer function
        """
        return tf(self.denominator, self.numerator, -self.deadtime)

    def step(self, *args):
        """ Step response """ 
        return signal.lti(self.numerator, self.denominator).step(*args)

    def simplify(self):
        g = polygcd(self.numerator, self.denominator)
        self.numerator, remainder = self.numerator/g
        self.denominator, remainder = self.denominator/g
    
    def __repr__(self):
        if self.name:
            r = str(self.name) + "\n"
        else:
            r = ''
        r += "tf(" + str(self.numerator.coeffs) + ", " + str(self.denominator.coeffs)
        if self.deadtime != 0:
            r += ", deadtime=" + str(self.deadtime)
        if self.u: 
            r += ", u='" + self.u + "'"
        if self.y: 
            r += ", y=': " + self.y + "'"
        r += ")"
        return r

    def __call__(self, s):
        """
        This allows the transfer function to be evaluated at
        particular values of s.
        Effectively, this makes a tf object behave just like a function of s.

        >>> G = tf(1, [1, 1])
        >>> G(0)
        1.0
        """
        return (numpy.polyval(self.numerator, s) /
                numpy.polyval(self.denominator, s) *
                numpy.exp(-s * self.deadtime))

    def __add__(self, other):
        if not isinstance(other, tf):
            other = tf(other)
        if self.deadtime != other.deadtime:
            raise ValueError("Transfer functions can only be added if their deadtimes are the same")
        gcd = self.denominator * other.denominator
        return tf(self.numerator*other.denominator +
                  other.numerator*self.denominator, gcd, self.deadtime)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if not isinstance(other, tf):
            other = tf(other)
        return tf(self.numerator*other.numerator,
                  self.denominator*other.denominator,
                  self.deadtime + other.deadtime)

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        if not isinstance(other, tf):
            other = tf(other)
        return self * other.inverse()

    def __rdiv__(self, other):
        return tf(other)/self

    def __neg__(self):
        return tf(-self.numerator, self.denominator, self.deadtime)

    def __pow__(self, other):
        r = self
        for k in range(other-1):
            r = r * self
        return r

def feedback(forward, backward=None, positive=False):
    """
    Defined for use in connect function
    Calculates a feedback loop
    This version is for trasnfer function objects
    Negative feedback is assumed, use positive=True for positive feedback
    Forward refers to the function that goes out of the comparator
    Backward refers to the function that goes into the comparator
    """

    # Create identity tf if no backward defined
    if backward is None:
        backward = 1
    if positive:
        backward = -backward
    return  forward * 1/(1 + backward * forward)


def tf_step(tf, t_final=10, initial_val=0, steps=100):
    """
    Prints the step response of a transfer function
    """
    # See the following docs for meaning of *args
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.step.html
    
    # Surpress the complex casting error
    import warnings
    warnings.simplefilter("ignore")
    # TODO: Make more specific
    
    deadtime = tf.deadtime
    tspace = numpy.linspace(0, t_final, steps)
    foo = numpy.real(tf.step(initial_val, tspace))
    t_stepsize = max(foo[0])/(foo[0].size-1)
    t_startindex = int(max(0, numpy.round(deadtime/t_stepsize, 0)))
    foo[1] = numpy.roll(foo[1], t_startindex)
    foo[1,0:t_startindex] = initial_val
    plt.plot(foo[0], foo[1])
    plt.show()

# TODO: Concatenate tf objects into MIMO structure


def sigmas(A):
    """
    Returns the singular values of A
    
    Parameters
    ----------
    A : array
        Transfer function matrix.
        
    Returns
    -------
    :math:`\sigma` (A) : array
        Singular values of A arranged in decending order.

    This is a convenience wrapper to enable easy calculation of
    singular values over frequency

    Example
    -------
    
    >>> A = numpy.array([[1, 2],
    ...                  [3, 4]])
    >>> sigmas(A)
    array([ 5.4649857 ,  0.36596619])

    """
    #TODO: This should probably be created with functools.partial
    return numpy.linalg.svd(A, compute_uv=False)


def sv_dir(G, table=False):
    """
    Returns the input and output singular vectors associated with the
    minimum and maximum singular values.
       
    Parameters
    ----------
    G : array of complex numbers
        Transfer function matrix.
    
    table : True of False boolean
            Default set to False.
            
    Returns
    -------
    u : list of arrays containing complex numbers
        Output vector associated with the maximum and minium singular
        values. The maximum singular output vector is the first entry u[0] and
        the minimum is the second u[1].
    
    v : list of arrays containing complex numbers
        Input vector associated with the maximum and minium singular
        values. The maximum singular intput vector is the first entry u[0] and
        the minimum is the second u[1].
    
    table : If table is True then the output and input vectors are summarised
            and returned as a table in the command window. Values are reported
            to five significant figures.
        
    NOTE
    ----
    If G is evaluated at a pole, u[0] is the input and v[0] is the output
    directions associated with that pole, respectively.
    
    If G is evaluated at a zero, u[1] is the input and v[1] is the output
    directions associated with that zero.    
    
    """
    U, Sv, V = SVD(G)
    
    u = [U[:, 0]] + [U[:, -1]]
    v = [V[:, 0]] + [V[:, -1]]


    if table == True:
        Headings = ['Maximum', 'Minimum']

        for i in range(2):
            print(' ')
            print('Directions of %s SV' % Headings[i])
            print '-' * 24
            
            print('Output vector')
            for k in range(len(u[i])):  #change to len of u[i]
                print('%.5f %+.5fi' % (u[i][k].real, u[i][k].imag))
            print('Input vector')
            for k in range(len(v[i])):
                print('%.5f %+.5fi' % (v[i][k].real, v[i][k].imag))
                
            print(' ')
    
    return(u, v)


def SVD(Gin):
    """
    Returns the singular values (Sv) as well as the input and output
    singular vectors (V and U respectively).   
    
    Parameters
    ----------
    Gin : matrix of complex numbers
        Transfer function matrix.
    
    Returns
    -------
    U : matrix of complex numbers
        Unitary matrix of output singular vectors.
        
    Sv : array
        Singular values of `Gin` arranged in decending order.
        
    V : matrix of complex numbers
        Unitary matrix of input singular vectors. 
    
    NOTE
    ----
    `SVD(G) = U Sv VH`  where `VH` is the complex conjugate transpose of `V`.
    Here we return `V` and not `VH`.

    This is a convenience wrapper to enable easy calculation of
    singular values and their associated singular vectors as in Skogestad.
    
    """
    U, Sv, VH = numpy.linalg.svd(Gin)
    V = numpy.conj(numpy.transpose(VH))
    return(U, Sv, V)
 

def Wp(wB, A, s):
    """
    Computes the magnitude of the performance weighting function 
    as a function of s => `|Wp(s)|`.
    
    Parameters
    ----------
    wB : flaot
         Minimum bandwidth frequency requirment.
    
    A : float
        Maximum steady state tracking error.
    
    s : complex 
        Typically `w*1j`.
        
    Returns
    -------
    `|Wp(s)|` : float
        The magnitude of the performance weighting fucntion at a specific frequency (s).
        
    NOTE
    ----
    This is based on Skogestad eq 2.105 and is just one example of a performance weighting function.
    
    """
    M = 2
    return(numpy.abs((s/M + wB) / (s + wB*A)))     



def perf_Wp(S, wB_req, maxSSerror, wStart, wEnd):
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
    ...     return(K*numpy.exp(-t1*s)/(t2*s + 1))
    ... 
    >>> def L(s):
    ...     return(Kc*G(s))
    ... 
    >>> def S(s):
    ...     return(numpy.linalg.inv((numpy.eye(2) + L(s))))      #SVD of S = 1/(I + L)
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
        U, Sv, V = SVD(S(s[i]))
        magPlotS1[i] = Sv[0]
        magPlotS3[i] = Sv[-1]
        if (f < 1 and magPlotS1[i] > 0.707):
            wB = w[i]
            f = 1
    for i in range(len(w)):
        Wpi[i] = Wp(wB_req, maxSSerror, s[i])                                      
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
    plt.show()
    return(wB) 



def distRej(G, gd):
    """
    Convenience wrapper for calculation of ||gd||2, and the 
    disturbace condition number for each disturbance in your Gd matrix.
    
    Parameters
    ----------    
    G : matrix of complex numbers
        System transfer function matrix.
    
    gd : Vector of complex numbers
        Single disturbance vector (gdi) from your disturbance matrix Gd.
        
    Returns
    -------
    1/||gd|| :math:`_2` : float
        The inverse of the 2-norm of a single disturbance gd.
    
    Disturbance Condition Number : float
        The disturbance condition number :math:`\sigma` (G) :math:`\sigma` (G :math:`^{-1}` yd)
    
    """
    
    gd1 = 1/numpy.linalg.norm((gd), 2)   #Returns largest sing value of gd(wj)
    yd = gd1*gd
    distCondNum = sigmas(G)[0] * sigmas(numpy.linalg.inv(G)*yd)[0]
    return(gd1, distCondNum)



def MIMOnyqPlot(L, axLim, wStart, wEnd):
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
    ...     return(K*numpy.exp(-t1*s)/(t2*s + 1))
    ... 
    >>> def L(s):
    ...     return(Kc*G(s))
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
    y_up = numpy.sqrt(1-(x)**2)
    y_down = -1*numpy.sqrt(1 - (x)**2)
    plt.plot(x, y_up, 'b:', x, y_down, 'b:', lw=2)
    plt.plot(0, 0, 'r*', ms=10)
    plt.grid(True)
    n = axLim           # Sets x-axis limits
    plt.axis('equal')   # Ensure the unit circle remains round on resizing the figure
    plt.axis([-n, n, -n, n])
    fig = plt.gcf()
    BG = fig.patch
    BG.set_facecolor('white')
    plt.show()


   
def feedback_mimo(forward, backward=None, positive=False):
    """
    Calculates a feedback loop
    This version is for matrices
    Negative feedback is assumed, use positive=True for positive feedback
    Forward refers to the function that goes out of the comparator
    Backward refers to the function that goes into the comparator
    """

    # Create identity matrix if no backward matrix is specified
    if backward is None:
        backward = numpy.asmatrix(numpy.eye(numpy.shape(forward)[0],
                                  numpy.shape(forward)[1]))
    # Check the dimensions of the input matrices
    if backward.shape[1] != forward.shape[0]:
        raise ValueError("The column dimension of backward matrix must equal row dimension of forward matrix")
    forward = numpy.asmatrix(forward)
    backward = numpy.asmatrix(backward)
    I = numpy.asmatrix(numpy.eye(numpy.shape(backward)[0],
                                 numpy.shape(forward)[1]))
    if positive:
        backward = -backward
    return forward * (I + backward * forward).I


def omega(w_start, w_end):  
    """
    Convenience wrapper
    Defines the frequency range for calculation of frequency response
    Frequency in rad/time where time is the time unit used in the model.
    """
    omega = numpy.logspace(w_start, w_end, 1000)
    return omega
    
    
def freq(G):
    """ 
    Calculate the frequency response for an optimisation problem
    
    Parameters
    ----------
    G : tf
        plant model 
          
    Returns
    -------
    Gw : frequency response function           
    """ 

    def Gw(w):
        return G(1j*w)
    return Gw


def ControllerTuning(G, method='ZN'):
    """ 
    Calculates either the Ziegler-Nichols or Tyreus-Luyben
    tuning parameters for a PI controller based on the continuous
    cycling method.
    
    Parameters
    ----------
    G : tf
        plant model
    
    method : Use 'ZN' for Ziegler-Nichols tuning parameters and
             'TT' for Tyreus-Luyben parameters. The default is to
             return Ziegler-Nichols tuning parameters.
          
    Returns
    -------
    Kc : array containing a real number         
        proportional gain
    Taui : array containing a real number
        integral gain
    Ku : array containing a real number
        ultimate P controller gain
    Pu : array containing a real number
        corresponding period of oscillations                   
    """  
    
    settings = {'ZN' : [0.45, 0.83], 'TT' : [0.31, 2.2]}   
    
    GM, PM, wc, w_180 = margins(G)  
    Ku = numpy.abs(1 / G(1j * w_180))
    Pu = numpy.abs(2 * numpy.pi / w_180)
    Kc = Ku * settings.get(method)[0]
    Taui = Pu * settings.get(method)[1]

    return Kc, Taui, Ku, Pu


def margins(G):
    """ 
    Calculates the gain and phase margins, together with the gain and phase
    crossover frequency for a plant model
    
    Parameters
    ----------
    G : tf
        plant model         
          
    Returns
    -------    
    GM : array containing a real number
        gain margin
    PM : array containing a real number
        phase margin
    wc : array containing a real number
        gain crossover frequency where |G(jwc)| = 1
    w_180 : array containing a real number
        phase crossover frequency where angle[G(jw_180] = -180 deg
    """

    Gw = freq(G)

    def mod(x):
        """to give the function to calculate |G(jw)| = 1"""
        return numpy.abs(Gw(x)) - 1

    # how to calculate the freqeuncy at which |G(jw)| = 1
    wc = optimize.fsolve(mod, 0.1)

    def arg(w):
        """function to calculate the phase angle at -180 deg"""
        return numpy.angle(Gw(w)) + numpy.pi

    # where the freqeuncy is calculated where arg G(jw) = -180 deg
    w_180 = optimize.fsolve(arg, wc)

    PM = numpy.angle(Gw(wc), deg=True) + 180
    GM = 1/(numpy.abs(Gw(w_180)))

    return GM, PM, wc, w_180


def marginsclosedloop(L):
    """ 
    Calculates the gain and phase margins, together with the gain and phase
    crossover frequency for a control model
    
    Parameters
    ----------
    L : tf
        loop transfer function        
          
    Returns
    -------
    GM : real      
        gain margin
    PM : real           
        phase margin
    wc : real           
        gain crossover frequency for L
    wb : real           
        closed loop bandwidth for S
    wbt : real 
        closed loop bandwidth for T                  
    """
    
    GM, PM, wc, w_180 = margins(L)      
    S = feedback(1, L)
    T = feedback(L, 1)   
        
    Sw = freq(S)
    Tw = freq(T)
    
    def modS(x):
        return numpy.abs(Sw(x)) - 1/numpy.sqrt(2)
        
    def modT(x):
        return numpy.abs(Tw(x)) - 1/numpy.sqrt(2)        

    # calculate the freqeuncy at |S(jw)| = 0.707 from below (start searching from 0)
    wb = optimize.fsolve(modS, 0)  
    # calculate the freqeuncy at |T(jw)| = 0.707 from above (start searching from 1)
    wbt = optimize.fsolve(modT, 1) 

    #"Frequency range wb < wc < wbt    
    if (PM < 90) and (wb < wc) and (wc < wbt):
        valid = True
    else: valid = False
    return GM, PM, wc, wb, wbt, valid


def bode(G, w1, w2, label='Figure', margin=False):
    """ 
    Shows the bode plot for a plant model
    
    Parameters
    ----------
    G : tf
        plant transfer function
    w1 : real
        start frequency
    w2 : real
        end frequency
    label : string
        title for the figure (optional)
    margin : boolean
        show the cross over frequencies on the plot (optional)        
          
    Returns
    -------
    GM : array containing a real number      
        gain margin
    PM : array containing a real number           
        phase margin         
    """

    GM, PM, wc, w_180 = margins(G)

    # plotting of Bode plot and with corresponding frequencies for PM and GM
#    if ((w2 < numpy.log(w_180)) and margin):
#        w2 = numpy.log(w_180)  
    w = numpy.logspace(w1, w2, 1000)
    s = 1j*w

    plt.figure(label)
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
    phaseangle = phase(G(s), deg=True)
    plt.semilogx(w, phaseangle)
    if margin:
        plt.axvline(wc, color='black')
        plt.text(wc, numpy.average([numpy.max(phaseangle), numpy.min(phaseangle)]), '|G(jw)| = 1')
    plt.axhline(-180., color='red')
    plt.grid()
    plt.ylabel('Phase')
    plt.xlabel('Frequency [rad/unit time]')
    
    plt.show()

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
    S = feedback(1, L)
    T = feedback(L, 1)
    
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
    plt.semilogx(w, phase(L, deg=True))
    plt.semilogx(w, phase(S, deg=True))
    plt.semilogx(w, phase(T, deg=True))
    plt.grid()
    plt.ylabel("Phase")
    plt.xlabel("Frequency [rad/s]")  
    
    plt.show()



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
    ...     return(K*numpy.exp(-t1*s)/(t2*s + 1.))
    >>>
    >>> def Kc(s):
    ...     return(numpy.array([[0.1, 0.],
    ...                         [0., 0.1]])*10.)
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
        Sv1[i] = sigmas(Gin(s[i]))[0]
        Sv2[i] = sigmas(Gin(s[i]))[-1]
        if (f < 1 and Sv2[i] < 1):
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
    plt.show()
    
    if Kin is None:
        Bandwidth = wC
        print('Bandwidth = wC')
    else:
        def S(s):
            L = Kin(s)*Gin(s)
            dim1, dim2 = numpy.shape(Gin(0))
            return(numpy.linalg.inv(numpy.eye(dim1) + L))      #SVD of S = 1/(I + L)
        w = numpy.logspace(wStart, wEnd, 1000)
        s = w*1j
        Sv1 = numpy.zeros(len(w), dtype=complex)
        Sv2 = numpy.zeros(len(w), dtype=complex)
        f = 0
        wB = 0
        for i in range(len(w)):
            Sv1[i] = sigmas(S(s[i]))[0]
            Sv2[i] = sigmas(S(s[i]))[-1]
            if (f < 1 and Sv1[i] > 0.707):
                wB = w[i]
                f = 1
        plt.figure('MIMO Bode')
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
        plt.show()
        Bandwidth = wC, wB
        print('Bandwidth is a tuple of wC, wB')
    return(Bandwidth)

       

# according to convention this procedure should stay at the bottom       
if __name__ == '__main__':
    import doctest
    doctest.testmod()       
