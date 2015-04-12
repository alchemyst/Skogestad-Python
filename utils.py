# -*- coding: utf-8 -*-
'''
Created on Jan 27, 2012

@author: Carl Sandrock
'''

import numpy #do not abbreviate this module as np in utils.py
import sympy #do not abbreviate this module as sp in utils.py
from scipy import optimize, signal
import scipy.linalg as sc_linalg


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
        

def circle(cx, cy, r):
    """ 
    Return the coordinates of a circle
    
    Parameters
    ----------
    cx : float
        Center x coordinate.
        
    cy : float
        Center x coordinate.
        
    r : float
        Radius.
    
    Returns
    -------
    x, y : float
        Circle coordinates.
        
   """
    npoints = 100
    theta = numpy.linspace(0, 2*numpy.pi, npoints)
    y = cy + numpy.sin(theta)*r
    x = cx + numpy.cos(theta)*r
    return x, y


def arrayfun(f, A):
    """ 
    Recurses down to scalar elements in A, then applies f, returning lists
    containing the result.
    
    Parameters
    ----------
    A : array
    f : function
    
    Returns
    -------
    arrayfun : list
        
   """
    if len(A.shape) == 0:
        return f(A)
    else:
        return [arrayfun(f, b) for b in A]


def listify(A):
    """ 
    Transform a gain value into a transfer function.
    
    Parameters
    ----------
    K : float
        Gain.
    
    Returns
    -------
    gaintf : tf
        Transfer function.
        
   """
    return [A]


def gaintf(K):
    """ 
    Transform a gain value into a transfer function.
    
    Parameters
    ----------
    K : float
        Gain.
    
    Returns
    -------
    gaintf : tf
        Transfer function.
        
   """
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
    """ 
    Return the phase angle in degrees or radians
    
    Parameters
    ----------
    G : tf
        Plant of transfer functions.
    deg : booleans
        True if radians result is required, otherwise degree is default
        (optional).
    
    Returns
    -------
    phase : float
        Phase angle.
        
   """
    return numpy.unwrap(numpy.angle(G, deg=deg), 
                        discont=180 if deg else numpy.pi)


def Closed_loop(Kz, Kp, Gz, Gp):
    """ 
    Return zero and pole polynomial for a closed loop function.
    
    Parameters
    ----------
    Kz & Gz : list
        Polynomial constants in the numerator.
    Kz & Gz : list
        Polynomial constants in the denominator.
    
    Returns
    -------
    Zeros_poly : list
        List of zero polynomial for closed loop function.
        
    Poles_poly : list
        List of pole polynomial for closed loop function.
        
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


def RGAnumber(G, I):
    """ 
    Computes the RGA (Relative Gain Array) number of a matrix.
    
    Parameters
    ----------
    G : numpy matrix
        Transfer function matrix.
        
    I : numpy matrix
        Pairing matrix.
        
    Returns
    -------
    RGA number : float
        RGA number.

    """    
    return numpy.sum(numpy.abs(RGA(G) - I))
    

def RGA(Gin):
    """ 
    Computes the RGA (Relative Gain Array) of a matrix.
    
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
    

def tf_step(G, t_end=10, initial_val=0, points=1000, constraint=None, Y=None, method='numeric'):
    """
    Validate the step response data of a transfer function by considering dead
    time and constraints. A unit step response is generated.  
    
    Parameters
    ----------
    G : tf
        Transfer function (input[u] or output[y]) to evauate step response.
        
    Y : tf
        Transfer function output[y] to evaluate constrain step response (optional)(required if constraint is specified).
        
    t_end : integer
        length of time to evaluate step response (optional).
    
    initial_val : integer
        starting value to evalaute step response (optional).
        
    points : integer
        number of iteration that will be calculated (optional).
        
    constraint : real
        The upper limit the step response cannot exceed. Is only calculated
        if a value is specified (optional).
        
    method : ['numeric','analytic']
        The method that is used to calculate a constrainted response. A
        constraint value is required (optional).
          
    Returns
    -------
    timedata : array
        Array of floating time values.  
        
    process : array (1 or 2 dim)
        1 or 2 dimensional array of floating process values.
    """ 
    # Surpress the complex casting error
    import warnings
    warnings.simplefilter("ignore")
    
    timedata = numpy.linspace(0, t_end, points)    
    
    if constraint is None:
        deadtime =G.deadtime        
        [timedata, processdata] = numpy.real(G.step(initial_val, timedata))
        t_stepsize = max(timedata)/(timedata.size-1)
        t_startindex = int(max(0, numpy.round(deadtime/t_stepsize, 0)))
        processdata = numpy.roll(processdata, t_startindex)
        processdata[0:t_startindex] = initial_val
        
    else:
        if method == 'numeric':
            A1, B1, C1, D1 = signal.tf2ss(G.numerator, G.denominator)
            #adjust the shape for complex state space functions
            x1 = numpy.zeros((numpy.shape(A1)[1], numpy.shape(B1)[1]))
            
            if constraint is not None:
                A2, B2, C2, D2 = signal.tf2ss(Y.numerator, Y.denominator)
                x2 = numpy.zeros((numpy.shape(A2)[1], numpy.shape(B2)[1]))
            
            dt = timedata[1]
            processdata1 = []
            processdata2 = []
            bconst = False
            u = 1
            
            for t in timedata:
                dxdt1 = A1*x1 + B1*u
                y1 = C1*x1 + D1*u
                
                if constraint is not None:
                    if (y1[0,0] > constraint) or bconst:
                        y1[0,0] = constraint  
                        bconst = True # once constraint the system is oversaturated
                        u = 0 # TODO : incorrect, find the correct switching condition
                    dxdt2 = A2*x2 + B2*u
                    y2 = C2*x2 + D2*u
                    x2 = x2 + dxdt2 * dt      
                    processdata2.append(y2[0,0])
                  
                x1 = x1 + dxdt1 * dt                
                processdata1.append(y1[0,0])
            if constraint:
                processdata = [processdata1, processdata2]
            else: processdata = processdata1
        elif method == 'analytic':
            # TODO: caluate intercept of step and constraint line
            timedata, processdata = [0,0]
        else: print 'Invalid function parameters'
        
    # TODO: calculate time response
    return timedata, processdata

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


    if table:
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
    
    return u, v


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
    return U, Sv, V
 

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
    return numpy.abs((s/M + wB) / (s + wB*A))


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
    
    gd1 = 1/numpy.linalg.norm(gd, 2)   #Returns largest sing value of gd(wj)
    yd = gd1*gd
    distCondNum = sigmas(G)[0] * sigmas(numpy.linalg.inv(G)*yd)[0]
    return gd1, distCondNum

   
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


###############################################################################
#                                Chapter 4                                    #
###############################################################################


def state_controllability(A, B):
    '''
    This method checks if the state space description of the system is state
    controllable according to Definition 4.1 (p127).
    
    Parameters
    ----------
    A : numpy matrix
        Matrix A of state-space representation.
    B : numpy matrix
        Matrix B of state-space representation.

    Returns
    -------
    state_control : boolean
        True if state controllable
    u_p : array
        Input pole vectors for the states u_p_i
    control_matrix : numpy matrix
        State Controllability Matrix
        
    Note
    ----
    This does not check for state controllability for systems with repeated
    poles.
    '''
    
    state_control = True

    A = numpy.asmatrix(A)
    B = numpy.asmatrix(B)
        
    # Compute all input pole vectors.
    ev, vl = sc_linalg.eig(A, left=True, right=False)
    u_p = []
    for i in range(vl.shape[1]):
        vli = numpy.asmatrix(vl[:,i]) 
        u_p.append(B.H*vli.T) 
    state_control = not any(numpy.linalg.norm(x) == 0.0 for x in u_p)

    # compute the controllability matrix
    c_plus = [A**n*B for n in range(A.shape[1])]
    control_matrix = numpy.hstack(c_plus)

    return state_control, u_p, control_matrix
    
    
def poles(G):
    '''
    Return the poles of a multivariable transfer function system. Applies
    Theorem 4.4 (p135).
    
    Parameters
    ----------
    G : transfer function matrix (numpy/sympy)
        A n x n plant matrix.

    Returns
    -------
    zero : array
        List of zeros.
        
    Example
    -------
    >>> def G(s):
    >>> return 1 / (s + 2) * sp.Matrix([[s - 1,  4],
    ...                                [4.5, 2 * (s - 1)]])
    >>> zero(G)
    [4.00000000000000]
    
    Note
    ----
    Not applicable for a non-squared plant, yet.
    '''
    
    s = sympy.Symbol('s')
    G = sympy.Matrix(G(s)) #convert to sympy matrix object
    det = sympy.simplify(G.det())
    pole = sympy.solve(sympy.denom(det))
    return pole 


def zeros(G=None, A=None, B=None, C=None, D=None):
    '''
    Return the zeros of a multivariable transfer function system for with
    transfer functions or state-space. For transfer functions, Theorem 4.5
    (p139) is used. For state-space, the method from Equations 4.66 and 4.67
    (p138) is applied.
    
    Parameters
    ----------
    G : transfer function matrix (numpy/sympy)
        A n x n plant matrix        
    A, B, C, D : numpy matrix
        State space parameters

    Returns
    -------
    pole : array
        List of poles.
        
    Example
    -------
    >>> def G(s):
    >>> return 1 / (s + 2) * sp.Matrix([[s - 1,  4],
    ...                                [4.5, 2 * (s - 1)]])
    >>> zero(G)
    [-2.00000000000000]
    
    Note
    ----
    Not applicable for a non-squared plant, yet. It is assumed that B,C,D will
    have values if A is defined.
    '''
    # TODO create a beter function to accept paramters and switch between tf and ss
    
    if not G is None:
        s = sympy.Symbol('s')
        G = sympy.Matrix(G(s)) #convert to sympy matrix object
        det = sympy.simplify(G.det())
        zero = sympy.solve(sympy.numer(det))
    
    elif not A is None:
        z = sympy.Symbol('z')
        top = numpy.hstack((A,B))
        bot = numpy.hstack((C,D))
        m = numpy.vstack((top, bot))
        M = numpy.Matrix(m)
        [rowsA, colsA] = numpy.shape(A)
        [rowsB, colsB] = numpy.shape(B)
        [rowsC, colsC] = numpy.shape(C)
        [rowsD, colsD] = numpy.shape(D)
        p1 = numpy.eye(rowsA)
        p2 = numpy.zeros((rowsB, colsB))
        p3 = numpy.zeros((rowsC, colsC))
        p4 = numpy.zeros((rowsD, colsD))
        top = numpy.hstack((p1, p2))
        bot = numpy.hstack((p3, p4))
        p = numpy.vstack((top, bot))
        Ig = sympy.Matrix(p)
        zIg = z * Ig
        f = zIg - M
        zf = f.det()
        zero = sympy.solve(zf, z)    
    
    return zero


def pole_zero_directions(G, vec, dir_type, display_type='a', e=0.00001):
    """
    Crude method to calculate the input and output direction of a pole or zero,
    from the SVD.
    
    Parameters
    ----------
    G : numpy matrix
        The transfer function G(s) of the system.
    vec : array
        A vector containing all the transmission poles or zeros of a system.
        
    dir_type : string
        Type of direction to calculate.  
            
        ==========     ============================
        dir_type       Choose
        ==========     ============================
        'p'            Poles
        'z'            Zeros
        ==========     ============================
        
    display_type : string
        Choose the type of directional data to return (optional).  
        
        ============   ============================
        display_type   Directional data to return
        ============   ============================
        'a'            All data (default)
        'u'            Only input direction
        'y'            Only output direction
        ============   ============================
    
    e : float
        Used in pole direction calculation, to avoid division by zero. Let
        epsilon be very small.
    
    Returns
    -------
    pz_dir : array
        Pole or zero direction in the form:
        (pole/zero, input direction, output direction)
        
    Note
    ----
    This method is going to give incorrect answers if the function G has pole
    zero cancellation. The proper method is to use the state-space.
    """
    
    if dir_type == 'p':
        dt = 0
    else:  # z
        dt = -1
        e = 0

    pz_dir = []
    for d in vec:
        g = G(d + e)

        U, _, V =  SVD(g)
        u = V[:,dt]
        y = U[:,dt]
        if display_type == 'u':
            pz_dir.append(u)
        elif display_type == 'y':
            pz_dir.append(y)
        else: # all data
            pz_dir.append((d, u, y))
        
    return pz_dir
    

# according to convention this procedure should stay at the bottom       
if __name__ == '__main__':
    import doctest
    doctest.testmod()       
