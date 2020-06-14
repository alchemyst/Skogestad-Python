# -*- coding: utf-8 -*-
"""
Created on Jan 27, 2012

@author: Carl Sandrock
"""
from __future__ import division
from __future__ import print_function
import numpy  # do not abbreviate this module as np in utils.py
import scipy
import sympy  # do not abbreviate this module as sp in utils.py
from scipy import optimize, signal
import scipy.linalg as sc_linalg
from functools import reduce
import itertools

from robustcontrol.InternalDelay import *

def astf(maybetf):
    """
    :param maybetf: something which could be a tf
    :return: a transfer function object

    >>> G = tf( 1, [ 1, 1])
    >>> astf(G)
    tf([1.], [1. 1.])

    >>> astf(1)
    tf([1.], [1.])

    >>> astf(numpy.matrix([[G, 1.], [0., G]]))
    matrix([[tf([1.], [1. 1.]), tf([1.], [1.])],
            [tf([0.], [1]), tf([1.], [1. 1.])]], dtype=object)

    """
    if isinstance(maybetf, (tf, mimotf)):
        return maybetf
    elif numpy.isscalar(maybetf):
        return tf(maybetf)
    else:  # Assume we have an array-like object
        return numpy.asmatrix(arrayfun(astf, numpy.asarray(maybetf)))


def polylatex(coefficients, variable='s'):
    """Return latex representation of a polynomial

    :param coefficients: iterable of coefficients in descending order
    :param variable: string containing variable to use

    """
    terms = []
    N = len(coefficients)
    for i, coefficient in enumerate(coefficients):
        if coefficient == 0:
            continue

        order = N - i - 1
        term = '{0:+}'.format(coefficient)
        if order >= 1:
            term += variable
        if order >= 2:
            term += '^{}'.format(order)

        terms.append(term)
    return "".join(terms).lstrip('+')


class tf(object):
    """
    Very basic transfer function object

    Construct with a numerator and denominator:

    >>> G = tf(1, [1, 1])
    >>> G
    tf([1.], [1. 1.])

    >>> G2 = tf(1, [2, 1])

    The object knows how to do:

    addition

    >>> G + G2
    tf([1.5 1. ], [1.  1.5 0.5])
    >>> G + G # check for simplification
    tf([2.], [1. 1.])

    multiplication

    >>> G * G2
    tf([0.5], [1.  1.5 0.5])

    division

    >>> G / G2
    tf([2. 1.], [1. 1.])

    Deadtime is supported:

    >>> G3 = tf(1, [1, 1], deadtime=2)
    >>> G3
    tf([1.], [1. 1.], deadtime=2)

    Note we can't add transfer functions with different deadtime:

    >>> G2 + G3
    Traceback (most recent call last):
        ...
    ValueError: Transfer functions can only be added if their deadtimes are the same. self=tf([0.5], [1.  0.5]), other=tf([1.], [1. 1.], deadtime=2)

    Although we can add a zero-gain tf to anything

    >>> G2 + 0*G3
    tf([0.5], [1.  0.5])

    >>> 0*G2 + G3
    tf([1.], [1. 1.], deadtime=2)


    It is sometimes useful to define

    >>> s = tf([1, 0])
    >>> 1 + s
    tf([1. 1.], [1.])

    >>> 1/(s + 1)
    tf([1.], [1. 1.])
    """

    def __init__(self, numerator, denominator=1, deadtime=0, name='',
                 u='', y='', prec=3):
        """
        Initialize the transfer function from a
        numerator and denominator polynomial
        """
        # TODO: poly1d should be replaced by np.polynomial.Polynomial
        self.numerator = numpy.poly1d(numerator)
        self.denominator = numpy.poly1d(denominator)
        self.deadtime = deadtime
        self.zerogain = False
        self.name = name
        self.u = u
        self.y = y
        self.simplify(dec=prec)

    def inverse(self):
        """
        Inverse of the transfer function
        """
        return tf(self.denominator, self.numerator, -self.deadtime)

    def step(self, *args, **kwargs):
        """ Step response """
        return signal.lti(self.numerator, self.denominator).step(*args, **kwargs)

    def lsim(self, *args, **kwargs):
        """ Negative step response """
        return signal.lsim(signal.lti(self.numerator, self.denominator), *args, **kwargs)

    def simplify(self, dec=3):

        # Polynomial simplification
        k = self.numerator[self.numerator.order] / self.denominator[self.denominator.order]
        ps = self.poles().tolist()
        zs = self.zeros().tolist()

        ps_to_canc_ind, zs_to_canc_ind = common_roots_ind(ps, zs)
        cancelled = cancel_by_ind(ps, ps_to_canc_ind)

        places = 10
        if cancelled > 0:
            cancel_by_ind(zs, zs_to_canc_ind)
            places = dec

        self.numerator = numpy.poly1d(
            [round(i.real, places) for i in k*numpy.poly1d(zs, True)])
        self.denominator = numpy.poly1d(
            [round(i.real, places) for i in 1*numpy.poly1d(ps, True)])

        # Zero-gain transfer functions are special.  They effectively have no
        # dead time and can be simplified to a unity denominator
        if self.numerator == numpy.poly1d([0]):
            self.zerogain = True
            self.deadtime = 0
            self.denominator = numpy.poly1d([1])

    def simplify_euclid(self):
        """
        Cancels GCD from both the numerator and denominator.
        Uses the Euclidean algorithm for polynomial gcd

        Doctest:
        >>> G1 = tf([1], [1, 1])
        >>> G2 = tf([1], [2, 1])
        >>> G3 = G1 * G2
        >>> G4 = G3 * G1
        >>> G5 = G4 / G1
        >>> G3
        tf([0.5], [1.  1.5 0.5])
        >>> G5
        tf([0.5], [1.  1.5 0.5])
        """
        def gcd_euclid(a, b):
            """
            Euclidean algorithm for calculating the polynomial gcd:
            https://en.wikipedia.org/wiki/Polynomial_greatest_common_divisor#Euclidean_algorithm
            :param a: numpy.poly1d object
            :param b: numpy.poly1d object
            :return: numpy.poly1d object that is the GCD of a and b
            """
            if a.order < b.order:
                return gcd_euclid(b, a)

            if b == numpy.poly1d(0):
                return a

            _, r = numpy.polydiv(a, b)
            return gcd_euclid(b, r)

        gcd = gcd_euclid(self.denominator, self.numerator)
        if gcd.order == 0:
            return
        self.numerator, _ = numpy.polydiv(self.numerator, gcd)
        self.denominator, _ = numpy.polydiv(self.denominator, gcd)

    def poles(self):
        return self.denominator.r

    def zeros(self):
        return self.numerator.r

    def exp(self):
        """ If this is basically "D*s" defined as tf([D, 0], 1),
            return dead time

        >>> s = tf([1, 0], 1)
        >>> numpy.exp(-2*s)
        tf([1.], [1.], deadtime=2.0)

        """
        # Check that denominator is 1:
        if self.denominator != numpy.poly1d([1]):
            raise ValueError(
                'Can only exponentiate multiples of s, not {}'.format(self))
        s = tf([1, 0], 1)
        ratio = -self/s

        if len(ratio.numerator.coeffs) != 1:
            raise ValueError(
                'Can not determine dead time associated with {}'.format(self))

        D = ratio.numerator.coeffs[0]

        return tf(1, 1, deadtime=D)

    def __repr__(self):
        if self.name:
            r = str(self.name) + "\n"
        else:
            r = ''
        r += "tf(" + str(self.numerator.coeffs) + ", " \
            + str(self.denominator.coeffs)
        if self.deadtime:
            r += ", deadtime=" + str(self.deadtime)
        if self.u:
            r += ", u='" + self.u + "'"
        if self.y:
            r += ", y=': " + self.y + "'"
        r += ")"
        return r

    def _repr_latex_(self):
        num = polylatex(self.numerator.coefficients)
        den = polylatex(self.denominator.coefficients)

        if self.deadtime > 0:
            dt = "e^{{-{}s}}".format(self.deadtime)
            if len(self.numerator.coefficients.nonzero()[0]) > 1:
                num = "({})".format(num)
        else:
            dt = ""

        return r"$$\frac{{{}{}}}{{{}}}$$".format(num, dt, den)

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
        other = astf(other)
        if isinstance(other, numpy.matrix):
            return other.__add__(self)
        # Zero-gain functions are special
        dterrormsg = "Transfer functions can only be added if " \
                     "their deadtimes are the same. self={}, other={}"
        if self.deadtime != other.deadtime and not (
                self.zerogain or other.zerogain):
            raise ValueError(dterrormsg.format(self, other))
        gcd = self.denominator * other.denominator
        return tf(self.numerator*other.denominator +
                  other.numerator*self.denominator, gcd, self.deadtime +
                  other.deadtime)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = astf(other)
        if isinstance(other, numpy.matrix):
            return numpy.dot(other, self)
        elif isinstance(other, mimotf):
            return mimotf(numpy.dot(other.matrix, self))
        return tf(self.numerator*other.numerator,
                  self.denominator*other.denominator,
                  self.deadtime + other.deadtime)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if not isinstance(other, tf):
            other = tf(other)
        return self * other.inverse()

    def __rtruediv__(self, other):
        return tf(other)/self

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
# TODO: Concatenate tf objects into MIMO structure


def RHPonly(x, round_precision=2):
    return list(
        set(numpy.round(xi, round_precision) for xi in x if xi.real > 0))


@numpy.vectorize
def evalfr(G, s):
    return G(s)


def matrix_as_scalar(M):
    """
    Return a scalar from a 1x1 matrix

    :param M: matrix
    :return: scalar part of matrix if it is 1x1 else just a matrix
    """
    if M.shape == (1, 1):
        return M[0, 0]
    else:
        return M


class mimotf(object):
    """ Represents MIMO transfer function matrix

    This is a pretty basic wrapper around the numpy.matrix class which deals
    with most of the heavy lifting.

    You can construct the object from siso tf objects similarly to calling
    numpy.matrix:

    >>> G11 = G12 = G21 = G22 = tf(1, [1, 1])
    >>> G = mimotf([[G11, G12], [G21, G22]])
    >>> G
    mimotf([[tf([1.], [1. 1.]) tf([1.], [1. 1.])]
     [tf([1.], [1. 1.]) tf([1.], [1. 1.])]])

    Some coersion will take place on the elements:
    >>> mimotf([[1]])
    mimotf([[tf([1.], [1.])]])

    The object knows how to do:

    addition

    >>> G + G
    mimotf([[tf([2.], [1. 1.]) tf([2.], [1. 1.])]
     [tf([2.], [1. 1.]) tf([2.], [1. 1.])]])

    >>> 0 + G
    mimotf([[tf([1.], [1. 1.]) tf([1.], [1. 1.])]
     [tf([1.], [1. 1.]) tf([1.], [1. 1.])]])

    >>> G + 0
    mimotf([[tf([1.], [1. 1.]) tf([1.], [1. 1.])]
     [tf([1.], [1. 1.]) tf([1.], [1. 1.])]])

    multiplication
    >>> G * G
    mimotf([[tf([2.], [1. 2. 1.]) tf([2.], [1. 2. 1.])]
     [tf([2.], [1. 2. 1.]) tf([2.], [1. 2. 1.])]])

    >>> 1*G
    mimotf([[tf([1.], [1. 1.]) tf([1.], [1. 1.])]
     [tf([1.], [1. 1.]) tf([1.], [1. 1.])]])

    >>> G*1
    mimotf([[tf([1.], [1. 1.]) tf([1.], [1. 1.])]
     [tf([1.], [1. 1.]) tf([1.], [1. 1.])]])

    >>> G*tf(1)
    mimotf([[tf([1.], [1. 1.]) tf([1.], [1. 1.])]
     [tf([1.], [1. 1.]) tf([1.], [1. 1.])]])

    >>> tf(1)*G
    mimotf([[tf([1.], [1. 1.]) tf([1.], [1. 1.])]
     [tf([1.], [1. 1.]) tf([1.], [1. 1.])]])

    exponentiation with positive integer constants

    >>> G**2
    mimotf([[tf([2.], [1. 2. 1.]) tf([2.], [1. 2. 1.])]
     [tf([2.], [1. 2. 1.]) tf([2.], [1. 2. 1.])]])

    """
    def __init__(self, matrix):
        # First coerce whatever we have into a matrix
        self.matrix = astf(numpy.asmatrix(matrix))
        # We only support matrices of transfer functions
        self.shape = self.matrix.shape

    def det(self):
        return det(self.matrix)

    def poles(self):
        """ Calculate poles
        >>> s = tf([1, 0], [1])
        >>> G = mimotf([[(s - 1) / (s + 2),  4 / (s + 2)],
        ...            [4.5 / (s + 2), 2 * (s - 1) / (s + 2)]])
        >>> G.poles()
        array([-2.])
        """
        return poles(self)

    def zeros(self):
        return zeros(self)

    def cofactor_mat(self):
        A = self.matrix
        m = A.shape[0]
        n = A.shape[1]
        C = numpy.zeros((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                minorij = det(
                    numpy.delete(numpy.delete(A, i, axis=0), j, axis=1))
                C[i, j] = (-1.)**(i+1+j+1)*minorij
        return C

    def inverse(self):
        """ Calculate inverse of mimotf object

        >>> s = tf([1, 0], 1)
        >>> G = mimotf([[(s - 1) / (s + 2),  4 / (s + 2)],
        ...              [4.5 / (s + 2), 2 * (s - 1) / (s + 2)]])
        >>> G.inverse()
        matrix([[tf([ 1. -1.], [ 1. -4.]), tf([-2.], [ 1. -4.])],
                [tf([-2.25], [ 1. -4.]), tf([ 0.5 -0.5], [ 1. -4.])]],
               dtype=object)

        >>> G.inverse()*G.matrix
        matrix([[tf([1.], [1.]), tf([0.], [1])],
                [tf([0.], [1]), tf([1.], [1.])]], dtype=object)

        """
        detA = det(self.matrix)
        C_T = self.cofactor_mat().T
        inv = (1./detA)*C_T
        return inv

    def step(self, u_input, t_start=0, t_end=100, points=1000):
        """
        Calculate the time domian step response of a mimotf object.
        
        Parameters:
        -----------
        
        u_input:    The values of the input variables to the transfer function.
                    The values can be given as a 1D list or 1D numpy.array 
                    object.
        t_start:    The lower bound of the time domain simulation.
        t_end:      The upper bound of the time domain simulation.
        points:     The number of iteration points used for the simulation.
        
        Returns:
        --------
        
        tspan:      The time values corresponding with the response data points.
        G_response: A list of arrays, with a length equal to the amount of 
                    outputs of the transfer function. The arrays contain the 
                    step response data of each output.
        """
        
        tspan = numpy.linspace(t_start, t_end, points)
        
        G_stepdata = [[0 for i in range(self.shape[0])], [0 for i in range(self.shape[1])]]
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if u_input[j] == 0:
                    G_stepdata[i][j] = numpy.zeros(tspan.shape)
                else:
                    G_stepdata[i][j] = tf_step(self[i, j], t_end = t_end)[1]
        
        G_response = []
        
        for i in range(2):
            G_response.append(sum(G_stepdata[i]))

        return tspan, G_response
    
    def __call__(self, s):
        """
        >>> G = mimotf([[1]])
        >>> G(0)
        matrix([[1.]])

        >>> firstorder= tf(1, [1, 1])
        >>> G = mimotf(firstorder)
        >>> G(0)
        matrix([[1.]])

        >>> G2 = mimotf([[firstorder]*2]*2)
        >>> G2(0)
        matrix([[1., 1.],
                [1., 1.]])
        """
        return evalfr(self.matrix, s)

    def __repr__(self):
        return "mimotf({})".format(str(self.matrix))

    def __add__(self, other):
        left = self.matrix
        if not isinstance(other, mimotf):
            if hasattr(other, 'shape'):
                right = mimotf(other).matrix
            else:
                right = tf(other)
        else:
            right = other.matrix
        return mimotf(left + right)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        left = matrix_as_scalar(self.matrix)
        if not isinstance(other, mimotf):
            other = mimotf(other)
        right = matrix_as_scalar(other.matrix)
        return mimotf(left*right)

    def __rmul__(self, other):
        right = matrix_as_scalar(self.matrix)
        if not isinstance(other, mimotf):
            other = mimotf(other)
        left = matrix_as_scalar(other.matrix)
        return mimotf(left*right)

    def __div__(self, other):
        raise NotImplemented("Division doesn't make sense on matrices")

    def __neg__(self):
        return mimotf(-self.matrix)

    def __pow__(self, other):
        r = self
        for k in range(other-1):
            r = r * self
        return r

    def __getitem__(self, item):
        result = mimotf(self.matrix.__getitem__(item))
        if result.shape == (1, 1):
            return result.matrix[0, 0]
        else:
            return result

    def __slice__(self, i, j):
        result = mimotf(self.matrix[i, j])
        if result.shape == (1, 1):
            return result.matrix[0, 0]
        else:
            return result


def scaling(G_hat, e, u, input_type='symbolic', Gd_hat=None, d=None):
    """
    Receives symbolic matrix of plant and disturbance transfer functions
    as well as array of maximum deviations, scales plant variables according
    to eq () and ()

    Parameters
    -----------
    G_hat       : matrix of plant WITHOUT deadtime
    e           : array of maximum plant output variable deviations
                  in same order as G matrix plant outputs
    u           : array of maximum plant input variable deviations
                  in same order as G matrix plant inputs
    input_type  : specifies whether input is symbolic matrix or utils mimotf
    Gd_hat      : optional
                  matrix of plant disturbance model WITHOUT deadtime
    d           : optional
                  array of maximum plant disturbance variable deviations
                  in same order as Gd matrix plant disturbances
    Returns
    ----------
    G_scaled   : scaled plant function
    Gd_scaled  : scaled plant disturbance function

    Example
    -------
    >>> s = sympy.Symbol("s")

    >>> G_hat = sympy.Matrix([[1/(s + 2), s/(s**2 - 1)],
    ...                       [5*s/(s - 1), 1/(s + 5)]])

    >>> e = numpy.array([1,2])
    >>> u = numpy.array([3,4])

    >>> scaling(G_hat,e,u,input_type='symbolic')
    Matrix([
    [  3.0/(s + 2), 4.0*s/(s**2 - 1)],
    [7.5*s/(s - 1),      2.0/(s + 5)]])

    """

    De = numpy.diag(e)
    De_inv = numpy.linalg.inv(De)
    Du = numpy.diag(u)

    if Gd_hat is not None and d is not None:
        Dd = numpy.diag(d)

    if input_type == 'symbolic':
        G_scaled = De_inv*(G_hat)*(Du)
        if Gd_hat is not None and d is not None:
            Dd = numpy.diag(d)
            Gd_scaled = De_inv*(Gd_hat)*(Dd)
            if G_hat.shape == (1, 1):
                return G_scaled[0, 0], Gd_scaled[0, 0]
            else:
                return G_scaled, Gd_scaled
        else:
            if G_hat.shape == (1, 1):
                return G_scaled[0, 0]
            else:
                return G_scaled

    elif input_type == 'mimotf':
        De_inv_utils = [[] for r in range(De_inv.shape[0])]
        Du_utils = [[] for r in range(Du.shape[0])]

        for r in range(De_inv.shape[0]):
            for c in range(De_inv.shape[1]):
                De_inv_utils[r].append(tf([De_inv[r, c]]))
        for r in range(Du.shape[0]):
            for c in range(Du.shape[1]):
                Du_utils[r].append(tf([Du[r, c]]))

        De_inv_mimo = mimotf(De_inv_utils)
        Du_mimo = mimotf(Du_utils)
        G_scaled = De_inv_mimo*(G_hat)*(Du_mimo)

        if Gd_hat is not None and d is not None:
            Dd_utils = [[] for r in range(Dd.shape[0])]
            for r in range(Dd.shape[0]):
                for c in range(Dd.shape[1]):
                    Dd_utils[r].append(tf([Dd[r, c]]))
            Dd_mimo = mimotf(Dd_utils)
            Gd_scaled = De_inv_mimo*(Gd_hat)*(Dd_mimo)
            if G_hat.shape == (1, 1):
                return G_scaled[0, 0], Gd_scaled[0, 0]
            else:
                return G_scaled, Gd_scaled
        else:
            if G_hat.shape == (1, 1):
                return G_scaled[0, 0]
            else:
                return G_scaled
    else:
        raise ValueError('No input type specified')


def tf_step(G, t_end=10, initial_val=0, points=1000,
            constraint=None, Y=None, method='numeric'):
    """
    Validate the step response data of a transfer function by considering dead
    time and constraints. A unit step response is generated.

    Parameters
    ----------
    G : tf
        Transfer function (input[u] or output[y]) to evauate step response.

    Y : tf
        Transfer function output[y] to evaluate constrain step response
        (optional) (required if constraint is specified).

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
        deadtime = G.deadtime
        [timedata, processdata] = numpy.real(G.step(initial_val, timedata))
        t_stepsize = max(timedata)/(timedata.size-1)
        t_startindex = int(max(0, numpy.round(deadtime/t_stepsize, 0)))
        processdata = numpy.roll(processdata, t_startindex)
        processdata[0:t_startindex] = initial_val

    else:
        if method == 'numeric':
            A1, B1, C1, D1 = signal.tf2ss(G.numerator, G.denominator)
            # adjust the shape for complex state space functions
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
                    if (y1[0, 0] > constraint) or bconst:
                        y1[0, 0] = constraint
                        # once constraint the system is oversaturated
                        bconst = True
                        # TODO : incorrect, find the correct switch condition
                        u = 0
                    dxdt2 = A2*x2 + B2*u
                    y2 = C2*x2 + D2*u
                    x2 = x2 + dxdt2 * dt
                    processdata2.append(y2[0, 0])

                x1 = x1 + dxdt1 * dt
                processdata1.append(y1[0, 0])
            if constraint:
                processdata = [processdata1, processdata2]
            else:
                processdata = processdata1
        elif method == 'analytic':
            # TODO: calculate intercept of step and constraint line
            timedata, processdata = [0, 0]
        else:
            raise ValueError('Invalid function parameters')

    # TODO: calculate time response
    return timedata, processdata


def circle(cx, cy, r):
    """
    Return the coordinates of a circle

    Parameters
    ----------
    cx : float
        Center x coordinate.

    cy : float
        Center y coordinate.

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


def common_roots_ind(a, b, dec=3):
    #Returns the indices of common (approximately equal) roots
    #of two polynomials
    a_ind = []  # Contains index of common roots
    b_ind = []

    for i in range(len(a)):
        for j in range(len(b)):
            if abs(a[i]-b[j]) < 10**-dec:
                if j not in b_ind:
                    b_ind.append(j)
                    a_ind.append(i)
                    break
    return a_ind, b_ind


def cancel_by_ind(a, a_ind):
    #Removes roots by index, returns number of roots
    #that have been removed
    cancelled = 0  # Number of roots cancelled
    for i in a_ind:
        del a[i - cancelled]
        cancelled += 1
    return cancelled

def polygcd(a, b):
    """
    Find the approximate Greatest Common Divisor of two polynomials

    >>> a = numpy.poly1d([1, 1]) * numpy.poly1d([1, 2])
    >>> b = numpy.poly1d([1, 1]) * numpy.poly1d([1, 3])
    >>> polygcd(a, b)
    poly1d([1., 1.])

    >>> polygcd(numpy.poly1d([1, 1]), numpy.poly1d([1]))
    poly1d([1.])
    """
    a_roots = a.r.tolist()
    b_roots = b.r.tolist()
    a_common, b_common = common_roots_ind(a_roots, b_roots)
    gcd_roots = []
    for i in range(len(a_common)):
        gcd_roots.append((a_roots[a_common[i]]+b_roots[b_common[i]])/2)
    return numpy.poly1d(gcd_roots, True)


def polylcm(a, b):
    #Finds the approximate lowest common multiple of
    #two polynomials

    a_roots = a.r.tolist()
    b_roots = b.r.tolist()
    a_common, b_common = common_roots_ind(a_roots, b_roots)
    cancelled = cancel_by_ind(a_roots, a_common)
    if cancelled > 0:    #some roots in common
        gcd = polygcd(a, b)
        cancelled = cancel_by_ind(b_roots, b_common)
        lcm_roots = a_roots + b_roots
        return numpy.polymul(gcd, numpy.poly1d(lcm_roots, True))
    else:    #no roots in common
        lcm_roots = a_roots + b_roots
        return numpy.poly1d(lcm_roots, True)

def multi_polylcm(P):
    roots_list = [i.r.tolist() for i in P]
    roots_by_mult = []
    lcm_roots_by_mult = []
    for roots in roots_list:
        root_builder = []
        for root in roots:
            repeated = False
            for i in range(len(root_builder)):
                if abs(root_builder[i][0]-root) < 10**-3:
                    root_builder[i][1] += 1
                    repeated = True
                    break
            if not repeated:
                root_builder.append([root, 1])
        for i in root_builder:
            roots_by_mult.append(i)
    for i in range(len(roots_by_mult)):
        in_lcm = False
        for j in range(len(lcm_roots_by_mult)):
            if abs(roots_by_mult[i][0] - lcm_roots_by_mult[j][0]) < 10**-3:
                in_lcm = True
                if lcm_roots_by_mult[j][1] < roots_by_mult[i][1]:
                    lcm_roots_by_mult[j][1] = roots_by_mult[i][1]
                break
        if not in_lcm:
            lcm_roots_by_mult.append(roots_by_mult[i])
    lcm_roots = []
    for i in lcm_roots_by_mult:
        for j in range(i[1]):
            lcm_roots.append(i[0])
    return numpy.poly1d(lcm_roots, True)

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

    >>> def f(x):
    ...     return 1.
    >>> arrayfun(f, numpy.array([1, 2, 3]))
    [1.0, 1.0, 1.0]

    >>> arrayfun(f, numpy.array([[1, 2, 3], [1, 2, 3]]))
    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

    >>> arrayfun(f, 1)
    1.0
    """
    if not hasattr(A, 'shape') or numpy.isscalar(A):
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


def det(A):
    """
    Calculate determinant via elementary operations

    :param A: Array-like object
    :return: determinant

    >>> det(2.)
    2.0

    >>> A = [[1., 2.],
    ...      [1., 2.]]
    >>> det(A)
    0.0

    >>> B = [[1., 2.],
    ...      [3., 4.]]
    >>> det(B)
    -2.0

    >>> C = [[1., 2., 3.],
    ...      [1., 3., 2.],
    ...      [3., 2., 1.]]
    >>> det(C)
    -12.0

    # Can handle matrices of tf objects
    >>> G11 = tf([1], [1, 2])
    >>> G = mimotf([[G11, G11], [G11, G11]])
    >>> det(G)
    tf([0.], [1])

    >>> G = mimotf([[G11, 2*G11], [G11**2, 3*G11]])
    >>> det(G)
    tf([ 3. 16. 28. 16.], [ 1. 10. 40. 80. 80. 32.])

    """
    if isinstance(A,tf) or isinstance(A,mimotf):
        A = A.matrix

    A = numpy.asmatrix(A)

    assert A.shape[0] == A.shape[1], "Matrix must be square for determinant " \
                                     "to exist"

    # Base case, if matrix is 1x1, return value
    if A.shape == (1, 1):
        return A[0, 0]

    # We expand by columns
    sign = 1
    result = 0
    cols = rows = list(range(A.shape[1]))
    for i in cols:
        submatrix = A[numpy.ix_(cols[1:], list(cols[:i]) + list(cols[i+1:]))]
        result += sign*A[0, i]*det(submatrix)
        sign *= -1

    return result

###############################################################################
#                                Chapter 2                                    #
###############################################################################


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


def feedback(forward, backward=None, positive=False):
    """
    Defined for use in connect function
    Calculates a feedback loop
    This version is for transfer function objects
    Negative feedback is assumed, use positive=True for positive feedback
    Forward refers to the function that goes out of the comparator
    Backward refers to the function that goes into the comparator
    """

    # Create identity tf if no backward defined
    if backward is None:
        backward = 1
    if positive:
        backward = -backward
    return forward * 1/(1 + backward * forward)


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

    settings = {'ZN': [0.45, 0.83], 'TT': [0.31, 2.2]}

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

    # calculate the freqeuncy at |S(jw)| = 0.707 from below (start from 0)
    wb = optimize.fsolve(modS, 0)
    # calculate the freqeuncy at |T(jw)| = 0.707 from above (start from 1)
    wbt = optimize.fsolve(modT, 1)

    # Frequency range wb < wc < wbt
    if (PM < 90) and (wb < wc) and (wc < wbt):
        valid = True
    else:
        valid = False
    return GM, PM, wc, wb, wbt, valid


def Wp(wB, M, A, s):
    """
    Computes the magnitude of the performance weighting function. Based on
    Equation 2.105 (p62).

    Parameters
    ----------
    wB : float
        Approximate bandwidth requirement. Asymptote crosses 1 at this
        frequency.
    M : float
        Maximum frequency.
    A : float
        Maximum steady state tracking error. Typically 0.
    s : complex
        Typically `w*1j`.

    Returns
    -------
    `|Wp(s)|` : float
        The magnitude of the performance weighting fucntion at a specific
        frequency (s).

    NOTE
    ----
    This is just one example of a performance weighting function.
    """

    return (s / M + wB) / (s + wB * A)


def maxpeak(G, w_start=-2, w_end=2, points=1000):
    """
    Computes the maximum bode magnitude peak of a transfer function
    """
    w = numpy.logspace(w_start, w_end, points)
    s = 1j*w

    M = numpy.max(numpy.abs(G(s)))

    return M

###############################################################################
#                                Chapter 3                                    #
###############################################################################


def sym2mimotf(Gmat, deadtime=None):
    """Converts a MIMO transfer function system in sympy.Matrix form to a
    mimotf object making use of individual tf objects.

    Parameters
    ----------
    Gmat : sympy matrix
           The system transfer function matrix.
    deadtime: numpy matrix of same dimensions as Gmat
              The dead times of Gmat with corresponding indexes.

    Returns
    -------
    Gmimotf : sympy matrix
              The mimotf system matrix

    Example
    -------
    >>> s = sympy.Symbol("s")

    >>> G = sympy.Matrix([[1/(s + 1), 1/(s + 2)],
    ...                   [1/(s + 3), 1/(s + 4)]])

    >>> deadtime = numpy.matrix([[1, 5], [0, 3]])

    >>> sym2mimotf(G, deadtime)
    mimotf([[tf([1.], [1. 1.], deadtime=1) tf([1.], [1. 2.], deadtime=5)]
     [tf([1.], [1. 3.]) tf([1.], [1. 4.], deadtime=3)]])

    """
    rows, cols = Gmat.shape
    # Create empty list of lists. Appended to form mimotf input list
    Gtf = [[] for y in range(rows)]
    # Checks matrix dimensions, create dummy zero matrix if not added
    if deadtime is None:
        DT = numpy.zeros(Gmat.shape)
    elif Gmat.shape != deadtime.shape:
        return  Exception("Matrix dimensions incompatible")
    else:
        DT = deadtime

    for i in range(rows):
        for j in range(cols):
            G = Gmat[i, j]

            # Select function denominator and convert to list of coefficients
            Gnum, Gden = G.as_numer_denom()

            def poly_coeffs(G_comp):
                if G_comp.is_Number:  # can't convert single value to Poly
                    G_comp_tf = float(G_comp)
                else:
                    G_comp_poly = sympy.Poly(G_comp)
                    G_comp_tf = [float(k) for k in G_comp_poly.all_coeffs()]
                return G_comp_tf

            Gtf_num = poly_coeffs(Gnum)
            Gtf_den = poly_coeffs(Gden)
            Gtf[i].append(tf(Gtf_num, Gtf_den, DT[i, j]))

    Gmimotf = mimotf(Gtf)
    return Gmimotf


def mimotf2sym(G, deadtime=False):
    """Converts a mimotf object making use of individual tf objects to a transfer function system in sympy.Matrix form.

    Parameters
    ----------
    G : mimotf matrix
        The mimotf system matrix.
    deadtime: boolean
              Should deadtime be added to sympy matrix or not.

    Returns
    -------
    Gs : sympy matrix
         The sympy system matrix
    s : sympy symbol
         Sympy symbol generated

    Example
    -------
    >>> G = mimotf([[tf([1.], [1., 1.], deadtime=1), tf([1.], [1., 2.], deadtime=5)],
    ...             [tf([1.], [1., 3.]), tf([1.], [1., 4.], deadtime=3)]])

    >>> mimotf2sym(G, deadtime=True)
    (Matrix([
    [1.0*exp(-s)/(1.0*s + 1.0), 1.0*exp(-5*s)/(1.0*s + 2.0)],
    [        1.0/(1.0*s + 3.0), 1.0*exp(-3*s)/(1.0*s + 4.0)]]), s)
    >>> mimotf2sym(G, deadtime=False)
    (Matrix([
    [1.0/(1.0*s + 1.0), 1.0/(1.0*s + 2.0)],
    [1.0/(1.0*s + 3.0), 1.0/(1.0*s + 4.0)]]), s)
    """

    s = sympy.Symbol("s")
    rows, cols = G.shape
    terms = []
    for tf in G.matrix.A1:
        num_poly = sympy.Poly(tf.numerator.coeffs, s)
        den_poly = sympy.Poly(tf.denominator.coeffs, s)
        if deadtime:
            terms.append(num_poly * sympy.exp(-tf.deadtime * s) / den_poly)
        else:
            terms.append(num_poly / den_poly)
    Gs = sympy.Matrix([terms]).reshape(rows, cols)
    return Gs, s


def RGAnumber(G, I):
    """
    Computes the RGA (Relative Gain Array) number of a matrix.

    Parameters
    ----------
    G : numpy matrix (n x n)
        The transfer function G(s) of the system.
    I : numpy matrix
        Pairing matrix.

    Returns
    -------
    RGA number : float
        RGA number.

    """
    return numpy.sum(numpy.abs(RGA(G) - I))


def RGA(G):
    """
    Computes the RGA (Relative Gain Array) of a matrix.

    Parameters
    ----------
    G : numpy matrix (n x n)
        The transfer function G(s) of the system.

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
    G = numpy.asmatrix(G).astype('float')
    G = numpy.asarray(G)
    Ginv = numpy.linalg.pinv(G)
    return G*Ginv.T


def IterRGA(A, n):
    """
    Computes the n'th iteration of the RGA.

    Parameters
    ----------
    G : numpy matrix (n x n)
        The transfer function G(s) of the system.

    Returns
    -------
    n'th iteration of RGA matrix : matrix
        iterated RGA matrix of complex numbers.

    Example
    -------
    >>> G = numpy.array([[1, 2], [-1, 1]])
    >>> IterRGA(G, 4).round(3)
    array([[-0.004,  1.004],
           [ 1.004, -0.004]])

    """
    for _ in range(0, n):
        A = RGA(A)
    return A


def sigmas(A, position=None):
    """
    Returns the singular values of A

    Parameters
    ----------
    A : array
        State space system matrix A.
    position : string
        Type of sigmas to return (optional).

        =========      ==================================
        position       Type of sigmas to return
        =========      ==================================
        max            Maximum singular value
        min            Minimal singular value
        =========      ==================================

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
    array([5.4649857 , 0.36596619])
    >>> print("{:0.6}".format(sigmas(A, 'min')))
    0.365966
    """

    sigmas = numpy.linalg.svd(A, compute_uv=False)
    if position is not None:
        if position == 'max':
            sigmas = sigmas[0]
        elif position == 'min':
            sigmas = sigmas[-1]
        else:
            raise ValueError('Incorrect position parameter')

    return sigmas


def sv_dir(G, table=False):
    """
    Returns the input and output singular vectors associated with the
    minimum and maximum singular values.

    Parameters
    ----------
    G : numpy matrix (n x n)
        The transfer function G(s) of the system.
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
            print('-' * 24)

            print('Output vector')
            for k in range(len(u[i])):  # change to len of u[i]
                print('%.5f %+.5fi' % (u[i][k].real, u[i][k].imag))
            print('Input vector')
            for k in range(len(v[i])):
                print('%.5f %+.5fi' % (v[i][k].real, v[i][k].imag))

            print(' ')

    return u, v


def SVD(G):
    """
    Returns the singular values (Sv) as well as the input and output
    singular vectors (V and U respectively).

    Parameters
    ----------
    G : numpy matrix (n x n)
        The transfer function G(s) of the system.

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
    U, Sv, VH = numpy.linalg.svd(G)
    V = numpy.conj(numpy.transpose(VH))
    return U, Sv, V


def feedback_mimo(G, K=None, positive=False):
    """
    Calculates a feedback loop transfer function, and returns it as a mimotf
    object.
    Currently functionality allows for a controller with proportional 
    gain. 
    
    Parameters:
    ------------
    
    G : The main process transfer function as a mimotf class object.
    
    K : The controller transfer function as a numpy.matrix object.
    
    Returns:
    --------
    
    G_fb : The transfer function representing the feedback loop as a mimotf
    object.
    
    """
    
    if K is None:
        K = numpy.eye(G.matrix.shape[0])
        
    L = mimotf(K*G.matrix)
    IG = numpy.eye(2) + L.matrix
    IG = mimotf(IG)
    Gi = L*IG.inverse()
    
    return Gi


###############################################################################
#                                Chapter 4                                    #
###############################################################################


def tf2ss(H):
    """
    Converts a mimotf object to the controllable canonical form state space
    representation. This method and the examples were obtained from course work
    notes available at
    http://www.egr.msu.edu/classes/me851/jchoi/lecture/Lect_20.pdf
    which appears to derive the method from "A Linear Systems Primer"
    by Antsaklis and Birkhauser.

    Parameters
    ----------
    H : mimotf
        The mimotf object transfer function form

    Returns
    -------
    Ac : numpy matrix
        The state matrix of the observable system
    Bc : nump matrix
        The input matrix of the observable system
    Cc : numpy matrix
        The output matrix of the observable system
    Dc : numpy matrix
        The output matrix of the observable system

    Example
    -------
    >>> H = mimotf([[tf([1,1],[1,2]),tf(1,[1,1])],
    ...             [tf(1,[1,1]),tf(1,[1,1])]])
    >>> Ac, Bc, Cc, Dc = tf2ss(H)
    >>> Ac
    matrix([[ 0.,  1.,  0.],
            [-2., -3.,  0.],
            [ 0.,  0., -1.]])
    >>> Bc
    matrix([[0., 0.],
            [1., 0.],
            [0., 1.]])
    >>> Cc
    matrix([[-1., -1.,  1.],
            [ 2.,  1.,  1.]])
    >>> Dc
    matrix([[1., 0.],
            [0., 0.]])

    # This example from the source material doesn't work as shown because the
    # common zero and pole in H11 get cancelled during simplification
    >>> H = mimotf([[tf([4, 7, 3], [1, 4, 5, 2])], [tf(1, [1, 1])]])
    >>> Ac, Bc, Cc, Dc = tf2ss(H)
    >>> Ac # doctest: +SKIP
    matrix([[ 0.,  1.,  0.,  0.],
            [ 0.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  1.],
            [-2., -7., -9., -5.],
    >>> Bc # doctest: +SKIP
    matrix([[ 0.],
            [ 0.],
            [ 0.],
            [ 1.]])
    >>> Cc # doctest: +SKIP
    matrix([[ 3., 10.,  11.,  4.],
            [ 2.,  5.,  4.,  1.]])
    >>> Dc # doctest: +SKIP
    matrix([[ 0.],
            [ 0.]])

    """

    p, m = H.shape
    d = [[] for k in range(m)]  # Construct some empty lists for use later
    mu = [[] for k in range(m)]
    Lvect = [[] for k in range(m)]
    Llist = []
    D = numpy.asmatrix(numpy.zeros((m, m),
                                   dtype=numpy.lib.polynomial.poly1d))
    Hinf = numpy.asmatrix(numpy.zeros((p, m)))

    for j in range(m):
        lcm = numpy.poly1d(1)
        for i in range(p):
            # Find the lcm of the denominators of the elements in each column
            lcm = polylcm(lcm, H[i, j].denominator)
            # Check if the individual elements are proper
            if H[i, j].numerator.order == H[i, j].denominator.order:
                # Approximate the limit as s->oo for the TF elements
                Hinf[i, j] = H[i, j].numerator.coeffs[0] / \
                    H[i, j].denominator.coeffs[0]
            elif H[i, j].numerator.order > H[i, j].denominator.order:
                raise ValueError('Enter a matrix of stricly proper TFs')

        d[j] = tf(lcm)  # Convert lcm to a tf object
        mu[j] = lcm.order
        D[j, j] = d[j]  # Create a diagonal matrix of lcms
        # Create list of coeffs of lcm for column, excl highest order element
        Lvect[j] = list((d[j].numerator.coeffs[1:]))
        Lvect[j].reverse()
        Llist.append(Lvect[j])  # Create block diag matrix from list of lists

    Lmat = numpy.asmatrix(sc_linalg.block_diag(*Llist))  # Convert L to matrix
    N = H*D
    MS = N - Hinf*D

    def num_coeffs(x):
        return x.numerator.coeffs

    def offdiag(m):
        return numpy.asmatrix(numpy.diag(numpy.ones(m-1), 1))

    def lowerdiag(m):
        vzeros = numpy.zeros((m, 1))
        vzeros[-1] = 1
        return vzeros

    MSrows, MScols = MS.shape
    # This loop generates the M matrix, which forms the output matrix, C
    Mlist = []
    for j in range(MScols):
        maxlength = max(len(num_coeffs(MS[k, j])) for k in range(MSrows))
        assert maxlength == mu[j]
        Mj = numpy.zeros((p, maxlength))
        for i in range(MSrows):
            M_coeffs = list(num_coeffs(MS[i, j]))
            M_coeffs.reverse()
            Mj[i, maxlength-len(M_coeffs):] = M_coeffs
        Mlist.append(Mj)
    Mmat = numpy.asmatrix(numpy.hstack(Mlist))

    # construct an off diagonal matrix used to form the state matrix
    Acbar = numpy.asmatrix(
        sc_linalg.block_diag(*[offdiag(order) for order in mu]))
    # construct a lower diagonal matrix which forms the input matrix
    Bcbar = numpy.asmatrix(
        sc_linalg.block_diag(*[lowerdiag(order) for order in mu]))

    Ac = Acbar - Bcbar*Lmat
    Bc = Bcbar
    Cc = Mmat
    Dc = Hinf

    return Ac, Bc, Cc, Dc


def state_controllability(A, B):
    """
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
    """

    state_control = True

    A = numpy.asmatrix(A)
    B = numpy.asmatrix(B)

    # Compute all input pole vectors.
    ev, vl = sc_linalg.eig(A, left=True, right=False)
    u_p = []
    for i in range(vl.shape[1]):
        vli = numpy.asmatrix(vl[:, i])
        u_p.append(B.H*vli.T)
    state_control = not any(numpy.linalg.norm(x) == 0.0 for x in u_p)

    # compute the controllability matrix
    c_plus = [A**n*B for n in range(A.shape[1])]
    control_matrix = numpy.hstack(c_plus)

    return state_control, u_p, control_matrix


def state_observability_matrix(a, c):
    """calculate the observability matrix

    :param a:  numpy matrix
              the A matrix in the state space model

    :param c: numpy matrix
              the C matrix in the state space model

    Example:
    --------

    >>> A = numpy.matrix([[0, 0, 0, 0],
    ...                   [0, -2, 0, 0],
    ...                   [2.5, 2.5, -1, 0],
    ...                   [2.5, 2.5, 0, -3]])

    >>> C = numpy.matrix([0, 0, 1, 1])

    >>> state_observability_matrix(A, C)
    matrix([[  0.,   0.,   1.,   1.],
            [  5.,   5.,  -1.,  -3.],
            [-10., -20.,   1.,   9.],
            [ 25.,  65.,  -1., -27.]])
    """

    # calculate the number of states
    n_states = numpy.shape(a)[0]

    # construct the observability matrix
    observability_m = [c*a**n for n in range(n_states)]
    observability_m = numpy.vstack(observability_m)

    return observability_m


def kalman_controllable(A, B, C, P=None, RP=None):
    """Computes the Kalman Controllable Canonical Form of the inout system
    A, B, C, making use of QR Decomposition. Can be used in sequentially with
    kalman_observable to obtain a minimal realisation.

    Parameters
    ----------
    A :  numpy matrix
         The system state matrix.
    B :  numpy matrix
         The system input matrix.
    C :  numpy matrix
         The system output matrix.
    P :  (optional) numpy matrix
         The controllability matrix
    RP : (optional int)

    Returns
    -------
    Ac : numpy matrix
         The state matrix of the controllable system
    Bc : nump matrix
         The input matrix of the controllable system
    Cc : numpy matrix
         The output matrix of the controllable system

    Example
    -------
    >>> A = numpy.matrix([[0, 0, 0, 0],
    ...                   [0, -2, 0, 0],
    ...                   [2.5, 2.5, -1, 0],
    ...                   [2.5, 2.5, 0, -3]])

    >>> B = numpy.matrix([[1],
    ...                   [1],
    ...                   [0],
    ...                   [0]])

    >>> C = numpy.matrix([0, 0, 1, 1])

    >>> Ac, Bc, Cc = kalman_controllable(A, B, C)
    >>> def round(A):
    ...     return numpy.round(A + 1e-5, 3)
    >>> round(Ac)
    array([[-1.   , -0.196,  0.189],
           [-5.099, -1.962, -0.999],
           [ 0.   , -0.999, -2.038]])

    >>> round(Bc)
    array([[-1.414],
           [ 0.   ],
           [ 0.   ]])

    >>> round(Cc)
    array([[0.   , 1.387, 0.053]])
    """
    nstates = A.shape[1]  # Compute the number of states

    #Calculate controllability matrix if necessary
    if P is None:
        _, _, P = state_controllability(A, B)

    # Find rank of the controllability matrix if necessary
    if RP is None:
        RP = numpy.linalg.matrix_rank(P)

    if RP == nstates:

        return A, B, C

    elif RP < nstates:
        # compute the QR decomposition of the controllability matrix
        T, R = numpy.linalg.qr(P)
        # separate the controllable subspace of T
        T1 = numpy.matrix(T[:, 0:RP])
        # Separate out the elements orthogonal to the controllable subspace
        T2 = numpy.matrix(T[:, RP:nstates])
        # Calculate the controllable state matrix
        Ac = T1.T*A*T1
        # Calculate the observable state matrix
        Bc = T1.T*B
        # Calculate the observable output matrix
        Cc = C*T1

        return Ac, Bc, Cc


def kalman_observable(A, B, C, Q=None, RQ=None):
    """Computes the Kalman Observable Canonical Form of the inout system
    A, B, C, making use of QR Decomposition. Can be used in sequentially
    with kalman_controllable to obtain a minimal realisation.

    Parameters
    ----------
    A :  numpy matrix
         The system state matrix.
    B :  numpy matrix
         The system input matrix.
    C :  numpy matrix
         The system output matrix.
    Q :  (optional) numpy matrix
         Observability matrix
    RQ : (optional) int
         Rank of observability matrxi

    Returns
    -------
    Ao : numpy matrix
        The state matrix of the observable system
    Bo : nump matrix
        The input matrix of the observable system
    Co : numpy matrix
        The output matrix of the observable system

    Example
    -------
    >>> A = numpy.matrix([[0, 0, 0, 0],
    ...                   [0, -2, 0, 0],
    ...                   [2.5, 2.5, -1, 0],
    ...                   [2.5, 2.5, 0, -3]])

    >>> B = numpy.matrix([[1],
    ...                   [1],
    ...                   [0],
    ...                   [0]])

    >>> C = numpy.matrix([0, 0, 1, 1])
    >>> Ao, Bo, Co = kalman_observable(A, B, C)
    >>> def round(A):
    ...     return numpy.round(A + 1e-5, 3)
    >>> round(Ao)
    array([[-2.   ,  5.099,  0.   ],
           [ 0.196, -1.038, -0.999],
           [ 0.189, -0.999, -0.962]])

    >>> round(Bo)
    array([[ 0.   ],
           [-1.387],
           [ 0.053]])

    >>> round(Co)
    array([[-1.414,  0.   ,  0.   ]])
    """
    nstates = A.shape[1]  # Compute the number of states
    # Compute the observability matrix if necessary
    if Q is None:
        Q = state_observability_matrix(A, C)

    # Compute rank of observability matrix if necessary
    if RQ is None:
        RQ = numpy.linalg.matrix_rank(Q)

    if RQ == nstates:

        return A, B, C
    # The system is not state observable
    elif RQ < nstates:
        # Compute the QR decomposition of the observability matrix
        V, R = numpy.linalg.qr(Q.T)
        # Separate out the elements of  the observable subspace
        V1 = V[:, 0:RQ]
        # Separate out the elements othrogonal to the observable subspace
        V2 = V[:, RQ:nstates]
        # Calculate the observable state matrix
        Ao = V1.T*A*V1
        # Calculate the observable input matrix
        Bo = V1.T*B
        # Calculate the observable output matrix
        Co = C*V1
        return Ao, Bo, Co


def minimal_realisation(a, b, c):
    """"This function will obtain a minimal realisation for a state space
    model in the form given in Skogestad second edition p 119 equations
    4.3 and 4.4

    :param a: numpy matrix
              the A matrix in the state space model

    :param b: numpy matrix
              the B matrix in the state space model

    :param c: numpy matrix
              the C matrix in the state space model


    Reference
    ---------
    The method (as well as the examples) are from
    https://ece.gmu.edu/~gbeale/ece_521/xmpl-521-kalman-min-real-01.pdf

    Examples
    --------

    Example 1:

    >>> A = numpy.matrix([[0, 0, 0, 0],
    ...                   [0, -2, 0, 0],
    ...                   [2.5, 2.5, -1, 0],
    ...                   [2.5, 2.5, 0, -3]])

    >>> B = numpy.matrix([[1],
    ...                   [1],
    ...                   [0],
    ...                   [0]])

    >>> C = numpy.matrix([0, 0, 1, 1])

    >>> Aco, Bco, Cco = minimal_realisation(A, B, C)

    Add null to eliminate negatives null elements (-0.)

    >>> Aco.round(decimals=3) + 0.
    array([[-2.038,  5.192],
           [ 0.377, -0.962]])

    >>> Bco.round(decimals=3) + 0.
    array([[ 0.   ],
           [-1.388]])

    >>> Cco.round(decimals=3) + 0.
    array([[-1.388,  0.   ]])

    Example 2:

    >>> A = numpy.matrix([[1, 1, 0],
    ...                    [0, 1, 0],
    ...                    [0, 1, 1]])

    >>> B = numpy.matrix([[0, 1],
    ...                   [1, 0],
    ...                   [0, 1]])

    >>> C = numpy.matrix([1, 1, 1])

    >>> Aco, Bco, Cco = minimal_realisation(A, B, C)

    Add null to eliminate negatives null elements (-0.)

    >>> Aco.round(decimals=3) + 0.
    array([[ 1.   ,  0.   ],
           [-1.414,  1.   ]])

    >>> Bco.round(decimals=3) + 0.
    array([[-1.   ,  0.   ],
           [ 0.   ,  1.414]])

    >>> Cco.round(decimals=3) + 0.
    array([[-1.   ,  1.414]])
    """
    # number of states
    n_states = numpy.shape(a)[0]

    # obtain the controllability matrix
    _, _, C = state_controllability(a, b)

    # obtain the observability matrix
    O = state_observability_matrix(a, c)

    # calculate the rank of the controllability and observability martix
    rank_C = numpy.linalg.matrix_rank(C)

    # transpose the observability matrix to calculate the column rank
    rank_O = numpy.linalg.matrix_rank(O.T)

    if rank_C <= rank_O and rank_C < n_states:
        Ac, Bc, Cc = kalman_controllable(
            a, b, c, P=C, RP=rank_C)

        ObsCont = state_observability_matrix(Ac, Cc)
        rank_ObsCont = numpy.linalg.matrix_rank(ObsCont.T)
        n_statesC = numpy.shape(Ac)[0]

        if rank_ObsCont < n_statesC:
            Aco, Bco, Cco = kalman_observable(
                Ac, Bc, Cc, Q=ObsCont, RQ=rank_ObsCont)
        else:
            return Ac, Bc, Cc

    elif rank_O < n_states:
        Ao, Bo, Co = kalman_observable(
            a, b, c, Q=O, RQ=rank_O)

        _, _, ContObs = state_controllability(Ao, Bo)
        rank_ContObs = numpy.linalg.matrix_rank(ContObs)
        n_statesO = numpy.shape(Ao)[0]

        if rank_ContObs < n_statesO:
            Aco, Bco, Cco = kalman_controllable(
                Ao, Bo, Co, P=ContObs, RP=rank_ContObs)
        else:
            return Ao, Bo, Co
    else:
        return a, b, c

    return Aco, Bco, Cco


def num_denom(A, symbolic_expr=False):

    sym_den = 0
    sym_num = 0
    s = sympy.Symbol('s')

    if isinstance(A,mimotf):
        denom = 1
        num = 1

        denom = [numpy.poly1d(denom) *
                 numpy.poly1d(A.matrix[0, j].denominator.coeffs)
                 for j in range(A.matrix.shape[1])]
        num = [numpy.poly1d(num) *
               numpy.poly1d(A.matrix[0, j].numerator.coeffs)
               for j in range(A.matrix.shape[1])]
        if symbolic_expr is True:
            for n in range(len(denom)):
                sym_den = (sym_den + denom[- n - 1] * s**n).simplify()
            for n in range(len(num)):
                sym_num = (sym_num + num[- n - 1] * s**n).simplify()
            return sym_num, sym_den
        else:
            return num, denom

    elif isinstance(A,tf):
        denom = []
        num = []

        denom = [list(A.denominator.coeffs)[n] for n in range(
            len(list(A.denominator.coeffs)))]
        num = [list(A.numerator.coeffs)[n] for n in range(
            len(list(A.numerator.coeffs)))]
        if symbolic_expr is True:
            for n in range(len(denom)):
                sym_den = (sym_den + denom[- n - 1] * s**n).simplify()
            for n in range(len(num)):
                sym_num = (sym_num + num[- n - 1] * s**n).simplify()
            return sym_num, sym_den
        else:
            return num, denom
"""
    else:
        sym_num, sym_den = A.as_numer_denom()
        if not symbolic_expr:
            num_poly   = sympy.Poly(sym_num)
            numer      = [float(k) for k in num_poly.all_coeffs()]
            den_poly   = sympy.Poly(sym_den)
            denom      = [float(k) for k in den_poly.all_coeffs()]
            return numer, denom
        else:
            return sym_num, sym_den
"""


def minors(G, order):
    """
    Returns the order minors of a MIMO tf G.
    """
    retlist = []
    Nrows, Ncols = G.shape
    for rowstokeep in itertools.combinations(range(Nrows), order):
        for colstokeep in itertools.combinations(range(Ncols), order):
            rowstokeep = numpy.array(rowstokeep)
            colstokeep = numpy.array(colstokeep)
            G_slice = G[rowstokeep[:, None], colstokeep]
            if isinstance(G_slice,tf):
                retlist.append(G_slice)
            elif (isinstance(G_slice,mimotf)) and (G_slice.shape[0] == G_slice.shape[1]):
                retlist.append(G_slice.det())
    return retlist


def lcm_of_all_minors(G):
    """
    Returns the lowest common multiple of all minors of G
    """
    Nrows, Ncols = G.shape
    denoms = []
    for i in range(1, min(Nrows, Ncols) + 1, 1):
        allminors = minors(G, i)
        for j in allminors:
            if j.denominator.order > 0:
                denoms.append(j.denominator)
    return multi_polylcm(denoms)


def poles_and_zeros_of_square_tf_matrix(G):
    """
    Determine poles and zeros of a square mimotf matrix, making use of the determinant.
    This method may fail in special cases. If terms cancel out during calculation of the determinant,
    not all poles and zeros will be determined.

    Parameters
    ----------
    G : mimotf matrix (n x n)
        The transfer function of the system.

    Returns
    -------
    z : array
        List of zeros.
    p : array
        List of poles.
    possible_cancel : boolean
                      Test whether terms were possibly cancelled out in determinant calculation.

    Example
    -------
    >>> G = mimotf([[tf([1, -1], [1, 2]), tf([4], [1, 2])],
    ...             [tf([4.5], [1, 2]), tf([2, -2], [1, 2])]])
    >>> poles_and_zeros_of_square_tf_matrix(G)
    (array([4.]), array([-2.]), False)
    """

    # Convert mimotf 2 sympy matrix
    Gs, s = mimotf2sym(G, deadtime=False)

    # Test cancellation
    rows, cols = G.shape
    r = Gs.rank()
    possible_cancel = False
    for i in range(rows):
        for j in range(cols):
            minor_multiply = Gs[r*i + j]
            g_minors = Gs.minor_submatrix(i, j)
            for minor in g_minors:
                numer_test = sympy.Poly(sympy.numer(minor_multiply).expand(), s)
                denom_test = sympy.Poly(sympy.denom(minor).expand(), s)
                _, res = sympy.div(numer_test, denom_test)
                if res == 0 and sympy.denom(minor) != 1:
                    possible_cancel = True

    # Determine determinant
    detG = Gs.det().simplify()

    # Determine numerator and denominator
    numer = sympy.numer(detG).expand()
    denom = sympy.denom(detG).expand()

    # Create numpy poly
    zero_poly = numpy.poly1d(sympy.Poly(numer, s).coeffs())
    pole_poly = numpy.poly1d(sympy.Poly(denom, s).coeffs())

    # Determine poly roots
    z = numpy.array(zero_poly.roots)
    p = numpy.array(pole_poly.roots)
    return z, p, possible_cancel


def poles(G=None, A=None):
    """
    If G is passed then return the poles of a multivariable transfer
    function system. Applies Theorem 4.4 (p135).
    If G is NOT specified but A is, returns the poles from
    the state space description as per section 4.4.2.

    Parameters
    ----------
    G : sympy or mimotf matrix (n x n)
        The transfer function G(s) of the system.
    A : State Space A matrix

    Returns
    -------
    pole : array
        List of poles.

    Example
    -------
    >>> s = tf([1,0],[1])
    >>> G = mimotf([[(s - 1) / (s + 2), 4 / (s + 2)],
    ...             [4.5 / (s + 2), 2 * (s - 1) / (s + 2)]])
    >>> poles(G)
    array([-2.])
    >>> A = numpy.matrix([[1,0,0],[0,8,0],[0,0,5]])
    >>> Poles = poles(None, A)
    """

    if G:
        if not (isinstance(G,tf)  or isinstance(G,mimotf)):
            G = sym2mimotf(G)
        lcm = lcm_of_all_minors(G)
        return lcm.r
    else:
        pole, _ = numpy.linalg.eig(A)
        return pole


def zeros(G=None, A=None, B=None, C=None, D=None):
    """
    Return the zeros of a multivariable transfer function system for with
    transfer functions or state-space. For transfer functions, Theorem 4.5
    (p139) is used. For state-space, the method from Equations 4.66 and 4.67
    (p138) is applied.

    Parameters
    ----------
    G : sympy or mimotf matrix (n x n)
        The transfer function G(s) of the system.
    A, B, C, D : numpy matrix
        State space parameters

    Returns
    -------
    zero : array
           List of zeros.

    Example
    -------
    >>> s = tf([1,0],[1])
    >>> G = mimotf([[(s - 1) / (s + 2), 4 / (s + 2)],
    ...             [4.5 / (s + 2), 2 * (s - 1) / (s + 2)]])
    >>> zeros(G)
    [4.0]

    Note
    ----
    Not applicable for a non-squared plant, yet. It is assumed that B,C,D will
    have values if A is defined.
    """
    # TODO create a beter function to accept parameters and
    # switch between tf and ss

    if G:
        if not (isinstance(G,tf) or isinstance(G,mimotf)):
            G = sym2mimotf(G)
        lcm = lcm_of_all_minors(G)
        allminors = minors(G, G.shape[0])
        gcd = None
        s = sympy.Symbol('s')
        for m in allminors:
            numer, denom = num_denom(m, symbolic_expr=True)
            if denom != lcm:
                numer *= denom
            if numer.find(s):
                num_coeff = [float(k) for k in numer.as_poly().all_coeffs()]
                if not gcd:
                    gcd = numpy.poly1d(num_coeff)
                else:
                    gcd = polygcd(gcd, numpy.poly1d(num_coeff))
            else:
                gcd = numpy.poly1d(numer)
        zero = list(set(numpy.roots(gcd)))
        pole = poles(G)
        for i in pole:
            if i in zero:
                zero.remove(i)
        return zero

    elif A is not None:
        M = numpy.bmat([[A, B],
                        [C, D]])
        Ig = numpy.zeros_like(M)
        d = numpy.arange(A.shape[0])
        Ig[d, d] = 1
        eigvals = scipy.linalg.eigvals(M, Ig)
        return eigvals[numpy.isfinite(eigvals) & (eigvals != 0)]
        # TODO: Check if there are any cases where we need the symbolic method:
        # z = sympy.Symbol('z')
        # Ig = sympy.Matrix(Ig)
        # return sympy.solve((M - z*Ig).det(), z)


def pole_zero_directions(G, vec, dir_type, display_type='a', e=1E-8, z_tol=1E-4, p_tol=1E-4):
    """
    Crude method to calculate the input and output direction of a pole or zero,
    from the SVD.

    Parameters
    ----------
    G : numpy matrix (n x n)
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
        Avoid division by zero. Let epsilon be very small (optional).

    z_tol : float
        Acceptable tolerance for zero validation. Let z_tol be small (optional).

    p_tol : float
        Acceptable tolerance for pole validation. Let p_tol be small (optional).

    Returns
    -------
    pz_dir : array
        Pole or zero direction in the form:
        (pole/zero, input direction, output direction, valid)

    valid : integer array
        If 1 the directions are valid, else if 0 the directions are not valid.

    Note
    ----
    This method is going to give incorrect answers if the function G has pole
    zero cancellation. The proper method is to use the state-space.

    The validity of the directions is determined by checking that the dot
    product of the two vectors is equal to the product of their norms. Another
    method is to work out element-wise ratios and see if they are all the same.
    """

    if dir_type == 'p':
        dt = 0
    elif dir_type == 'z':
        dt = -1
        e = 0
    else:
        raise ValueError('Incorrect dir_type parameter')

    N = len(vec)
    if display_type == 'a':
        pz_dir = []
    else:
        pz_dir = numpy.matrix(numpy.zeros([G(e).shape[0], N]))
        valid = []

    for i in range(N):
        d = vec[i]
        g = G(d + e)

        U, _, V = SVD(g)
        u = V[:, dt]
        y = U[:, dt]

        # validation
        v = False

        # zero validation test
        if dir_type == 'z':
            z_test = max(abs((g*u*y.I).A1))
            if z_test <= z_tol:
                v = True

        # pole validation test
        if dir_type == 'p':
            p_test = max(abs((y*u.I*g.I).A1))
            if p_test <= p_tol:
                v = True

        if display_type == 'u':
            pz_dir[:, i] = u
            valid.append(v)
        elif display_type == 'y':
            pz_dir[:, i] = y
            valid.append(v)
        elif display_type == 'a':
            pz_dir.append((d, u, y, [v]))
        else:
            raise ValueError('Incorrect display_type parameter')

    if display_type == 'a':
        display = pz_dir
    else:
        display = pz_dir, valid

    return display

def zero_directions_ss(A, B, C, D):
    """
    This function calculates the zeros with input and output directions from
    a state space representation using the method outlined on pg. 140

    Parameters
    ----------
    A : numpy matrix
        A matrix of state space representation
    B : numpy matrix
        B matrix of state space representation
    C : numpy matrix
        C matrix of state space representation
    D : numpy matrix
        D matrix of state space representation

    Returns
    -------
    zeros_in_out : list
        zeros_in_out[i] contains a zero, input direction vector and
        output direction vector
    """
    M = numpy.bmat([[A, B],
                    [C, D]])
    Ig = numpy.zeros_like(M)
    d = numpy.arange(A.shape[0])
    Ig[d, d] = 1

    eigvals_in, zeros_in = scipy.linalg.eig(M, b=Ig)
    eigvals_out, zeros_out = scipy.linalg.eig(M.T, b=Ig)

    #The input vector direction is given by u_z as shown in eq. 4.66.
    #The length of the vector is the same as the number of columns
    #in the B matrix.
    eigvals_in_dir = []
    for i in range(len(eigvals_in)):
        if numpy.isfinite(eigvals_in[i]) and eigvals_in[i] != 0:
            eigvals_in_dir.append([eigvals_in[i], zeros_in[-B.shape[1]:,i]])

    #Similar to the input vector
    #The length of the vector is the same as the number of rows
    #in the C matrix.
    eigvals_out_dir = []
    for i in range(len(eigvals_out)):
        if numpy.isfinite(eigvals_out[i]) and eigvals_out[i] != 0:
            eigvals_out_dir.append([eigvals_out[i], zeros_out[-C.shape[0]:,i]])

    #The eigenvalues are returned in no specific order. Sorting ensures
    #that the input and output directions get matched to the correct zero
    #value
    eigvals_in_dir.sort(key = lambda x: abs(x[0]))
    eigvals_out_dir.sort(key = lambda x: abs(x[0]))

    zeros_in_out = []
    for i in range(len(eigvals_in_dir)):
        zeros_in_out.append([eigvals_in_dir[i][0],
                             eigvals_in_dir[i][1],
                             eigvals_out_dir[i][1]])

    return zeros_in_out

###############################################################################
#                                Chapter 6                                    #
###############################################################################


def BoundST(G, poles, zeros, deadtime=None):
    """
    This function will calculate the minimum peak values of S and T if the
    system has zeros and poles in the input or output. For standard conditions
    Equation 6.8 (p224) is applied. Equation 6.16 (p226) is used when deadtime
    is included.

    Parameters
    ----------
    G : numpy matrix (n x n)
        The transfer function G(s) of the system.
    poles : numpy array (number of zeros)
        List of poles.
    zeros : numpy array (number of zeros)
        List of zeros.
    deadtime : numpy matrix (n, n)
        Deadtime or time delay for G.

    Returns
    -------
    Ms_min : real
        Minimum peak value.

    Note
    ----
    All the poles and zeros must be distict.
    """
    Np = len(poles)
    Nz = len(zeros)
    Yp, _ = pole_zero_directions(G, poles, 'p', 'y')
    Yz, _ = pole_zero_directions(G, zeros, 'z', 'y')

    yp_mat1 = numpy.matrix(
        numpy.diag(poles)) * numpy.matrix(numpy.ones([Np, Np]))
    yp_mat2 = yp_mat1.T
    Qp = (Yp.H * Yp) / (yp_mat1 + yp_mat2)

    yz_mat1 = (numpy.matrix(
        numpy.diag(zeros)) * numpy.matrix(numpy.ones([Nz, Nz])))
    yz_mat2 = yz_mat1.T
    Qz = (Yz.H * Yz) / (yz_mat1 + yz_mat2)

    yzp_mat1 = numpy.matrix(
        numpy.diag(zeros)) * numpy.matrix(numpy.ones([Nz, Np]))
    yzp_mat2 = numpy.matrix(
        numpy.ones([Nz, Np])) * numpy.matrix(numpy.diag(poles))
    Qzp = Yz.H * Yp / (yzp_mat1 - yzp_mat2)

    if deadtime is None:

        pre_mat = sc_linalg.sqrtm((numpy.linalg.inv(Qz))).dot(Qzp).dot(
            sc_linalg.sqrtm(numpy.linalg.inv(Qp)))
        # Final equation 6.8
        Ms_min = numpy.sqrt(1 + (numpy.max(sigmas(pre_mat))) ** 2)

    else:
        # Equation 6.16 (p226) uses maximum deadtime per output channel to
        # give tightest lowest bounds. Create vector to be used for the
        # diagonal deadtime matrix containing each outputs' maximum dead time.
        # This would ensure tighter bounds on T and S. The minimum function is
        # used because all stable systems have dead time with a negative sign.

        dead_time_vec_max_row = numpy.zeros(deadtime.shape[0])

        for i in range(deadtime.shape[0]):
            dead_time_vec_max_row[i] = numpy.max(abs(deadtime[i]))

        def Dead_time_matrix(s, dead_time_vec_max_row):
            dead_time_matrix = numpy.diag(numpy.exp(numpy.multiply(
                dead_time_vec_max_row, s)))
            return dead_time_matrix

        Q_dead = numpy.zeros((Np, Np))

        for i in range(Np):
            for j in range(Np):
                numerator_mat = (numpy.transpose(
                    numpy.conjugate(Yp[:, i])) * Dead_time_matrix(
                        poles[i], dead_time_vec_max_row) * Dead_time_matrix(
                            poles[j], dead_time_vec_max_row) * Yp[:, j])

                denominator_mat = poles[i] + poles[j]
                Q_dead[i, j] = numerator_mat / denominator_mat

        lambda_mat = sc_linalg.sqrtm(numpy.linalg.pinv(Q_dead)).dot(
            Qp + Qzp.dot(numpy.linalg.pinv(Qz)).dot(numpy.transpose(
                numpy.conjugate(Qzp)))).dot(sc_linalg.sqrtm(
                    numpy.linalg.pinv(Q_dead)))

        # Final equation 6.19
        Ms_min = float(numpy.real(numpy.max(numpy.linalg.eig(lambda_mat)[0])))

    return Ms_min


def BoundKS(G, poles, up, e=0.00001):
    """
    The function uses equaption 6.24 (p229) to calculate the peak value for KS
    transfer function using the stable version of the plant.

    Parameters
    ----------
    G : numpy matrix (n x n)
        The transfer function G(s) of the system.
    poles : numpy array (number of poles)
        List of right-half plane poles.
    up : numpy array (number of poles)
        List of input pole directions.
    e : float
        Avoid division by zero. Let epsilon be very small (optional).

    Returns
    -------
    KS_max : float
        Minimum peak value.
    """

    KS_PEAK = [numpy.linalg.norm(up.H * numpy.linalg.pinv(G(RHP_p + e)), 2)
               for RHP_p in poles]

    KS_max = numpy.max(KS_PEAK)

    return KS_max


def distRej(G, gd):
    """
    Convenience wrapper for calculation of ||gd||2 (equation 6.42, p238) and
    the disturbance condition number (equation 6.43) for each disturbance.

    Parameters
    ----------
    G : numpy matrix (n x n)
        The transfer function G(s) of the system.
    gd : numpy matrix (m x n)
        The transfer function Gd(s) of the distrurbances.

    Returns
    -------
    1/||gd|| :math:`_2` : float
        The inverse of the 2-norm of a single disturbance gd.

    yd : numpy matrix
        Disturbance direction.
        
    distCondNum : float
        The disturbance condition number
        :math:`\sigma` (G) :math:`\sigma` (G :math:`^{-1}` yd)
    
    """

    gd1 = 1 / numpy.linalg.norm(gd, 2)  # Returns largest sing value of gd(wj)
    yd = gd1 * gd
    distCondNum = sigmas(G)[0] * sigmas(numpy.linalg.inv(G) * yd)[0]

    return gd1, numpy.array(yd).reshape(-1,).tolist(), distCondNum


def distRHPZ(G, Gd, RHP_Z):
    """
    Applies equation 6.48 (p239) for performance requirements imposed by
    disturbances. Calculate the system's zeros alignment with the disturbacne
    matrix.

    Parameters
    ----------
    G : numpy matrix (n x n)
        The transfer function G(s) of the system.
    gd : numpy matrix (n x 1)
        The transfer function Gd(s) of the distrurbances.
    RHP_Z : complex
        Right-half plane zero

    Returns
    -------
    Dist_RHPZ : float
        Minimum peak value.

    Note
    ----
    The return value should be less than 1.
    """
    if numpy.real(RHP_Z) < 0:  # RHP-z
        raise ValueError('Function only applicable to RHP-zeros')
    Yz, _ = pole_zero_directions(G, [RHP_Z], 'z', 'y')
    Dist_RHPZ = numpy.abs(Yz.H * Gd(RHP_Z))[0, 0]

    return Dist_RHPZ

def ssr_solve(A, B, C, D):
    """
    Solves the zeros and poles of a state-space representation of a system.

    :param A: System state matrix
    :param B: matrix
    :param C: matrix
    :param D: matrix

    For information on the meanings of A, B, C, and D consult Skogestad 4.1.1

    Returns:
        zeros: The system's zeros
        poles: The system's poles

    TODO: Add any other relevant values to solve for, for example, if coprime
    factorisations are useful somewhere add them to this function's return
    dict rather than writing another function.
    """

    z = sympy.symbols('z')

    I_A = sympy.eye(A.shape[0])
    Z_B = sympy.zeros(B.shape[0])
    Z_C = sympy.zeros(C.shape[0])
    Z_D = sympy.zeros(D.shape[0])

    M = sympy.BlockMatrix([[A, B],
                           [C, D]])

    I_A = sympy.eye(A.shape[0])
    Ig = sympy.BlockMatrix(
        [[I_A, Z_B],
         [Z_C, Z_D]]
    )

    zIg = z * Ig
    P = sympy.Matrix(zIg - M)  # Equation 4.62, Section 4.5.1
    zf = P.det()
    ss_zeros = list(sympy.solve(zf, z))

    ss_poles = list(eig for eig, order in A.eigenvals().items())

    return ss_zeros, ss_poles

# according to convention this procedure should stay at the bottom
if __name__ == '__main__':
    import doctest
    import sys

    # Exit with an error code equal to number of failed tests
    sys.exit(doctest.testmod()[0])

