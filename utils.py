# -*- coding: utf-8 -*-
'''
Created on Jan 27, 2012

@author: Carl Sandrock
'''
from __future__ import division
from __future__ import print_function
import numpy  # do not abbreviate this module as np in utils.py
import scipy
import sympy  # do not abbreviate this module as sp in utils.py
from scipy import optimize, signal
import scipy.linalg as sc_linalg
import fractions
from decimal import Decimal
from functools import reduce
import itertools



def astf(maybetf):
    """
    :param maybetf: something which could be a tf
    :return: a transfer function object

    >>> G = tf(1, [1, 1])
    >>> astf(G)
    tf([ 1.], [ 1.  1.])

    >>> astf(1)
    tf([ 1.], [ 1.])

    >>> astf(numpy.matrix([[G, 1.], [0., G]]))
    matrix([[tf([ 1.], [ 1.  1.]), tf([ 1.], [ 1.])],
            [tf([ 0.], [1]), tf([ 1.], [ 1.  1.])]], dtype=object)

    """
    if isinstance(maybetf, (tf, mimotf)):
        return maybetf
    elif numpy.isscalar(maybetf):
        return tf(maybetf)
    else:  # Assume we have an array-like object
        return numpy.asmatrix(arrayfun(astf, numpy.asarray(maybetf)))


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
    ValueError: Transfer functions can only be added if their deadtimes are the same. self=tf([ 1.], [ 2.  1.]), other=tf([ 1.], [ 1.  1.], deadtime=2)

    Although we can add a zero-gain tf to anything

    >>> G2 + 0*G3
    tf([ 1.], [ 2.  1.])

    >>> 0*G2 + G3
    tf([ 1.], [ 1.  1.], deadtime=2)


    It is sometimes useful to define

    >>> s = tf([1, 0])
    >>> 1 + s
    tf([ 1.  1.], [ 1.])

    >>> 1/(s + 1)
    tf([ 1.], [ 1.  1.])
    """

    def __init__(self, numerator, denominator=1, deadtime=0, name='', u='', y='', prec=5):
        """
        Initialize the transfer function from a
        numerator and denominator polynomial
        """
        # TODO: poly1d should be replaced by np.polynomial.Polynomial
        self.numerator = numpy.poly1d(numerator)
        self.denominator = numpy.poly1d(denominator)
        self.deadtime = deadtime
        self.zerogain = False
        self.simplify(dec=prec)
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

    def simplify(self, dec=5):

        # Polynomial simplification
        g = polygcd(self.numerator, self.denominator)
        self.numerator, remainder = self.numerator/g
        assert numpy.allclose(remainder.coeffs, 0, atol=1e-6), "Error in simplifying rational, remainder=\n{}".format(remainder)
        self.denominator, remainder = self.denominator/g
        assert numpy.allclose(remainder.coeffs, 0, atol=1e-6), "Error in simplifying rational, remainder=\n{}".format(remainder)

        # Round numerator and denominator for coefficient simplification
        self.numerator = numpy.poly1d(numpy.round(self.numerator, dec))
        self.denominator = numpy.poly1d(numpy.round(self.denominator, dec))

        # Determine most digits in numerator & denominator
        num_dec = 0
        den_dec = 0
        for i in range(len(self.numerator.coeffs)):
            num_dec = max(num_dec, decimals(self.numerator.coeffs[i]))
        for j in range(len(self.denominator.coeffs)):
            den_dec = max(den_dec, decimals(self.denominator.coeffs[j]))

        # Convert coefficients to integers
        self.numerator = self.numerator*10**(max(num_dec, den_dec))
        self.denominator = self.denominator*10**(max(num_dec, den_dec))

        # Decimal-less representation of coefficients
        num_gcd = gcd(self.numerator.coeffs)
        den_gcd = gcd(self.denominator.coeffs)
        tf_gcd = gcd([num_gcd, den_gcd])
        self.numerator = self.numerator/tf_gcd
        self.denominator = self.denominator/tf_gcd

        # Zero-gain transfer functions are special.  They effectively have no
        # dead time and can be simplified to a unity denominator
        if self.numerator == numpy.poly1d([0]):
            self.zerogain = True
            self.deadtime = 0
            self.denominator = numpy.poly1d([1])

    def poles(self):
        return self.denominator.r

    def zeros(self):
        return self.numerator.r

    def exp(self):
        """ If this is basically "D*s" defined as tf([D, 0], 1),
            return dead time

        >>> s = tf([1, 0], 1)
        >>> numpy.exp(-2*s)
        tf([ 1.], [ 1.], deadtime=2.0)

        """
        # Check that denominator is 1:
        if self.denominator != numpy.poly1d([1]):
            raise ValueError('Can only exponentiate multiples of s, not {}'.format(self))
        s = tf([1, 0], 1)
        ratio = -self/s

        if len(ratio.numerator.coeffs) != 1:
            raise ValueError('Can not determine dead time associated with {}'.format(self))

        D = ratio.numerator.coeffs[0]

        return tf(1, 1, deadtime=D)

    def __repr__(self):
        if self.name:
            r = str(self.name) + "\n"
        else:
            r = ''
        r += "tf(" + str(self.numerator.coeffs) + ", " + str(self.denominator.coeffs)
        if self.deadtime:
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
                numpy.polyval(self.denominator, s) * numpy.exp(-s * self.deadtime))

    def __add__(self, other):
        other = astf(other)
        if isinstance(other, numpy.matrix):
            return other.__add__(self)
        # Zero-gain functions are special
        if self.deadtime != other.deadtime and not (self.zerogain or other.zerogain):
            raise ValueError("Transfer functions can only be added if their deadtimes are the same. self={}, other={}".format(self, other))
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
    mimotf([[tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  1.])]
     [tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  1.])]])

    Some coersion will take place on the elements:
    >>> mimotf([[1]])
    mimotf([[tf([ 1.], [ 1.])]])

    The object knows how to do:

    addition

    >>> G + G
    mimotf([[tf([ 2.], [ 1.  1.]) tf([ 2.], [ 1.  1.])]
     [tf([ 2.], [ 1.  1.]) tf([ 2.], [ 1.  1.])]])

    >>> 0 + G
    mimotf([[tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  1.])]
     [tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  1.])]])

    >>> G + 0
    mimotf([[tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  1.])]
     [tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  1.])]])

    multiplication
    >>> G * G
    mimotf([[tf([ 2.], [ 1.  2.  1.]) tf([ 2.], [ 1.  2.  1.])]
     [tf([ 2.], [ 1.  2.  1.]) tf([ 2.], [ 1.  2.  1.])]])

    >>> 1*G
    mimotf([[tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  1.])]
     [tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  1.])]])

    >>> G*1
    mimotf([[tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  1.])]
     [tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  1.])]])

    >>> G*tf(1)
    mimotf([[tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  1.])]
     [tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  1.])]])

    >>> tf(1)*G
    mimotf([[tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  1.])]
     [tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  1.])]])

    exponentiation with positive integer constants

    >>> G**2
    mimotf([[tf([ 2.], [ 1.  2.  1.]) tf([ 2.], [ 1.  2.  1.])]
     [tf([ 2.], [ 1.  2.  1.]) tf([ 2.], [ 1.  2.  1.])]])

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
        return self.det().poles()

    def zeros(self):
        return self.det().zeros()

    def cofactor_mat(self):
        A = self.matrix
        m = A.shape[0]
        n = A.shape[1]
        C = numpy.zeros((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                minorij = det(numpy.delete(numpy.delete(A, i, axis=0), j, axis=1))
                C[i, j] = (-1.)**(i+1+j+1)*minorij
        return C

    def inverse(self):
        """ Calculate inverse of mimotf object

        >>> s = tf([1, 0], 1)
        >>> G = mimotf([[(s - 1) / (s + 2),  4 / (s + 2)],
        ...              [4.5 / (s + 2), 2 * (s - 1) / (s + 2)]])
        >>> G.inverse()
        matrix([[tf([-1.  1.], [-1.  4.]), tf([ 2.], [-1.  4.])],
                [tf([ 9.], [ -4.  16.]), tf([-1.  1.], [-2.  8.])]], dtype=object)

        >>> G.inverse()*G.matrix
        matrix([[tf([ 1.], [ 1.]), tf([ 0.], [1])],
                [tf([ 0.], [1]), tf([ 1.], [ 1.])]], dtype=object)

        """
        detA = det(self.matrix)
        C_T = self.cofactor_mat().T
        inv = (1./detA)*C_T
        return inv

    def __call__(self, s):
        """
        >>> G = mimotf([[1]])
        >>> G(0)
        matrix([[ 1.]])

        >>> firstorder= tf(1, [1, 1])
        >>> G = mimotf(firstorder)
        >>> G(0)
        matrix([[ 1.]])

        >>> G2 = mimotf([[firstorder]*2]*2)
        >>> G2(0)
        matrix([[ 1.,  1.],
                [ 1.,  1.]])
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

    def __slice__(self, i, j):
        result = mimotf(self.matrix.__slice__(i, j))


def tf_step(G, t_end=10, initial_val=0, points=1000, constraint=None, Y=None, method='numeric'):
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
                        bconst = True  # once constraint the system is oversaturated
                        u = 0  # TODO : incorrect, find the correct switching condition
                    dxdt2 = A2*x2 + B2*u
                    y2 = C2*x2 + D2*u
                    x2 = x2 + dxdt2 * dt
                    processdata2.append(y2[0, 0])

                x1 = x1 + dxdt1 * dt
                processdata1.append(y1[0, 0])
            if constraint:
                processdata = [processdata1, processdata2]
            else: processdata = processdata1
        elif method == 'analytic':
            # TODO: calculate intercept of step and constraint line
            timedata, processdata = [0, 0]
        else: raise ValueError('Invalid function parameters')

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


def gcd(ar):
    return reduce(fractions.gcd, ar)


def decimals(fl):
    fl = str(fl)
    dec = abs(Decimal(fl).as_tuple().exponent)
    return dec


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
    # TODO: make this a little more natural (without the .matrix)
    >>> G11 = tf([1], [1, 2])
    >>> G = mimotf([[G11, G11], [G11, G11]])
    >>> det(G.matrix)
    tf([ 0.], [1])

    >>> G = mimotf([[G11, 2*G11], [G11**2, 3*G11]])
    >>> det(G.matrix)
    tf([  3.  16.  28.  16.], [  1.  10.  40.  80.  80.  32.])

    """

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
    return forward * 1/(1 + backward * forward)


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

    # Frequency range wb < wc < wbt
    if (PM < 90) and (wb < wc) and (wc < wbt):
        valid = True
    else: valid = False
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

def sym2mimotf(Gmat):
    """Converts a MIMO transfer function system in sympy.Matrix form to a mimotf object making use of individual tf objects.

    Parameters
    ----------
    Gmat : sympy matrix
           The system transfer function matrix.

    Returns
    -------
    Gmimotf : sympy matrix
              The mimotf system matrix

    Example
    -------
    >>> s = sympy.Symbol("s")

    >>> G = sympy.Matrix([[1/(s + 1), 1/(s + 2)],
    ...                   [1/(s + 3), 1/(s + 4)]])

    >>> sym2mimotf(G)
    mimotf([[tf([ 1.], [ 1.  1.]) tf([ 1.], [ 1.  2.])]
     [tf([ 1.], [ 1.  3.]) tf([ 1.], [ 1.  4.])]])

    """
    rows, cols = Gmat.shape
    #create empty list of lists. This will be appended to form mimotf input list
    Gtf=[[] for y in range(rows)]

    for i in range(rows):
        for j in range(cols):
            G = Gmat[i,j]
            #select function denominator and convert is to list of coefficients
            Gnum = G.as_numer_denom()[0]
            if Gnum.is_Number: # can't convert single value to Poly
                Gtf_num = float(Gnum)

            else:
                Gnum_poly = sympy.Poly(Gnum)
                Gtf_num = [float(k) for k in Gnum_poly.all_coeffs()]

            Gden = G.as_numer_denom()[1]
            if Gden.is_Number:
                Gtf_den = float(Gden)

            else:
                Gden_poly = sympy.Poly(Gden)
                Gtf_den = [float(k) for k in Gden_poly.all_coeffs()]
            Gtf[i].append(tf(Gtf_num,Gtf_den))
    Gmimotf = mimotf(Gtf)

    return Gmimotf


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
    G = numpy.asarray(G)
    Ginv = numpy.linalg.pinv(G)
    return G*Ginv.T


def sigmas(A, position=None):
    """
    Returns the singular values of A

    Parameters
    ----------
    A : array
        Transfer function matrix.
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
    array([ 5.4649857 ,  0.36596619])
    >>> print("{:0.6}".format(sigmas(A, 'min')))
    0.365966
    """

    sigmas = numpy.linalg.svd(A, compute_uv=False)
    if not position is None:
        if position == 'max':
            sigmas = sigmas[1]
        elif position == 'min':
            sigmas = sigmas[-1]
        else: raise ValueError('Incorrect position parameter')

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


def Kalman_controllable(A,B,C):
    """Computes the Kalman Controllable Canonical Form of the inout system A, B, C, making use of QR Decomposition.
       Can be used in sequentially with Kalman_observable to obtain a minimal realisation.
    Parameters
    ----------
    A : numpy matrix
        The system state matrix.
    B : numpy matrix
        The system input matrix.
    C : numpy matrix
        The system output matrix.
    rounding factor : integer
        The number of significant
    factor : int
        The number of additional significant digits after the first significant digit to round the returned matrix elements to.

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

    >>> Ac, Bc, Cc = Kalman_controllable(A, B, C)
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
    array([[ 0.   ,  1.387,  0.053]])
    """
    nstates = A.shape[1] #compute the number of states
    _, _, P = state_controllability(A,B) # compute the controllability matrix
    RP = numpy.linalg.matrix_rank(P) # find the rank of the controllability matrix

    if RP == nstates:

        return A,B,C

    elif RP < nstates:
        T, R = numpy.linalg.qr(P)# compute the QR decomposition of the controllability matrix
        T1 = numpy.matrix(T[:,0:RP]) # separate the controllable subspace of T
        T2 = numpy.matrix(T[:,RP:nstates])#separate out the elements orthogonal to the controllable subspace
        Ac = T1.T*A*T1#calculate the controllable state matrix
        Bc = T1.T*B#calculate the observable state matrix
        Cc = C*T1#calculate the observable output matrix

        return Ac,Bc,Cc


def Kalman_observable(A,B,C):
    """Computes the Kalman Observable Canonical Form of the inout system A, B, C, making use of QR Decomposition.
        Can be used in sequentially with Kalman_controllable to obtain a minimal realisation.

    Parameters
    ----------
    A : numpy matrix
        The system state matrix.
    B : numpy matrix
        The system input matrix.
    C : numpy matrix
        The system output matrix.
    rounding factor : integer
        The number of significant
    factor : int
        The number of additional significant digits after the first significant digit to round the returned matrix elements to.

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
    >>> Ao, Bo, Co = Kalman_observable(A, B, C)
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
    nstates = A.shape[1] #compute the number of states
    Q = state_observability_matrix(A,C)# compute the observability matrix
    RQ = numpy.linalg.matrix_rank(Q) # compute the rank of the observability matrix

    if RQ == nstates:

        return A, B, C

    elif RQ < nstates: #the system is not state observable
        V ,R = numpy.linalg.qr(Q.T) #compute the QR decomposition of the observability matrix
        V1 = V[:,0:RQ]# separate out the elements of  the observable subspace
        V2 = V[:,RQ:nstates] # separate out the elements othrogonal to the observable subspace
        Ao = V1.T*A*V1 # calculate the observable state matrix
        Bo = V1.T*B # calculate the observable input matrix
        Co = C*V1 # calculate the observable output matrix

        return Ao,Bo,Co


def remove_uncontrollable_or_unobservable_states(a, b, c, con_or_obs_matrix, uncontrollable=True, unobservable=False,
                                                 rank=None):
    """"remove the uncontrollable or unobservable states from the A, B and C state space matrices

    :param a: numpy matrix
              the A matrix in the state space model

    :param b: numpy matrix
              the B matrix in the state space model

    :param c: numpy matrix
              the C matrix in the state space model

    :param con_or_obs_matrix: numpy matrix
                              the controllable or observable matrix

    :param uncontrollable: boolean
                           set to True to remove uncontrollable states (default) or to false

    :param unobservable: boolean
                         set to True to remove unobservable states or to false (default)

    :param rank: optional (int)
                 rank of the controllable or observable matrix
                 if the rank is available set the rank=(rank of matrix) to avoid calculating matrix rank twice
                 by default rank=None and will be calculated

    Default: remove the uncontrollable states
    To remove the unobservable states set uncontrollable=False and unobservable=True

    return: the Kalman Canonical matrices
            Ac, Bc, Cc (the controllable subspace of A, B and C) if uncontrollable=True and unobservable=False
            or Ao, Bo, Co (the observable subspace of A, B and C) if uncontrollable=False and unobservable=True

    Note:
    If the controllable subspace of A, B and C are given (Ac, Bc and Cc) and the unobservable states are removed the
    matrices Aco, Bco and Cco (the controllable and observable subspace of A, B and C) will be returned

    If the observable subspace of A, B and C are given (Ao, Bo and Co) and the uncontrollable states are removed the
    matrices Aco, Bco and Cco (the controllable and observable subspace of A, B and C) will be returned

    Examples
    --------

    Example 1: remove uncontrollable states

    >>> A = numpy.matrix([[0, 0, 0, 0],
    ...                   [0, -2, 0, 0],
    ...                   [2.5, 2.5, -1, 0],
    ...                   [2.5, 2.5, 0, -3]])

    >>> B = numpy.matrix([[1],
    ...                   [1],
    ...                   [0],
    ...                   [0]])

    >>> C = numpy.matrix([0, 0, 1, 1])

    >>> controllability_matrix = numpy.matrix([[  1.,   0.,   0.,   0.],
    ...                                        [  1.,  -2.,   4.,  -8.],
    ...                                        [  0.,   5., -10.,  20.],
    ...                                        [  0.,   5., -20.,  70.]])

    >>> Ac, Bc, Cc = remove_uncontrollable_or_unobservable_states(A, B, C, controllability_matrix)

    Add null to eliminate negatives null elements (-0.)

    >>> Ac.round(decimals=3) + 0.
    array([[ 0.,  0.,  0.],
           [ 1.,  0., -6.],
           [ 0.,  1., -5.]])

    >>> Bc.round(decimals=3) + 0.
    array([[ 1.],
           [ 0.],
           [ 0.]])

    >>> Cc.round(decimals=3) + 0.
    array([[  0.,  10., -30.]])

    Example 2: remove unobservable states using Ac, Bc, Cc from example1

    >>> observability_matrix = numpy.matrix([[   0.,   10.,  -30.],
    ...                                      [  10.,  -30.,   90.],
    ...                                      [ -30.,   90., -270.]])

     >>> Ao, Bo, Co = remove_uncontrollable_or_unobservable_states(Ac, Bc, Cc, observability_matrix,
     ...                                                           uncontrollable=False, unobservable=True)

    >>> Ao.round(decimals=3) + 0.
    array([[ 0.,  1.],
           [ 0., -3.]])

    >>> Bo.round(decimals=3) + 0.
    array([[  0.],
           [ 10.]])

    >>> Co.round(decimals=3) + 0.
    array([[ 1.,  0.]])
    """

    # obtain the number of states
    n_states = numpy.shape(a)[0]

    # obtain matrix rank
    if rank is None:
        rank = numpy.linalg.matrix_rank(con_or_obs_matrix)

    # calculate the difference between the number of states and the number of controllable or observable states
    m = n_states - rank

    # if system is already state controllable or observable return matrices unchanged
    if m == 0:
        return a, b, c

    # create the a matrix P with dimensions n_states x n_states used to change matrices A, B and C to the
    # Kalman Canonical Form
    P = numpy.asmatrix(numpy.zeros((n_states, n_states)))

    if uncontrollable == True and unobservable == False:
        P[:, 0:rank] = con_or_obs_matrix[:, 0:rank]

        # this matrix will replace all the dependent columns in P to make P invertible
        replace_matrix = numpy.matrix(numpy.random.random((n_states, m)))

        # make P invertible
        P[:, rank:n_states] = replace_matrix

        # When removing the uncontrollable states the constructed matrix P is actually the inverse of P (P^-1) and
        # true matrix P is obtained by (P^-1)^-1
        P_inv = P
        P = numpy.linalg.inv(P_inv)

    elif uncontrollable == False and unobservable == True:
        P[0:rank, :] = con_or_obs_matrix[0:rank, :]

        # this matrix will replace all the dependent columns in P to make P invertible
        replace_matrix = numpy.matrix(numpy.random.random((m, n_states)))

        # make P invertible
        P[rank:n_states, :] = replace_matrix

        P_inv = numpy.linalg.inv(P)

    A_new = P*a*P_inv
    A_new = numpy.delete(A_new, numpy.s_[rank:n_states], 1)
    A_new = numpy.delete(A_new, numpy.s_[rank:n_states], 0)

    B_new = P*b
    B_new = numpy.delete(B_new, numpy.s_[rank:n_states], 0)

    C_new = c*P_inv
    C_new = numpy.delete(C_new, numpy.s_[rank:n_states], 1)

    return A_new, B_new, C_new


def minimal_realisation(a, b, c):
    """"This function will obtain a minimal realisation for a state space model in the form given in Skogestad
    second edition p 119 equations 4.3 and 4.4

    :param a: numpy matrix
              the A matrix in the state space model

    :param b: numpy matrix
              the B matrix in the state space model

    :param c: numpy matrix
              the C matrix in the state space model

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
    array([[ 0.,  1.],
           [ 0., -3.]])

    >>> Bco.round(decimals=3) + 0.
    array([[  0.],
           [ 10.]])

    >>> Cco.round(decimals=3) + 0.
    array([[ 1.,  0.]])

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
    array([[ 1.,  0.],
           [ 1.,  1.]])

    >>> Bco.round(decimals=3) + 0.
    array([[ 1.,  0.],
           [ 0.,  1.]])

    >>> Cco.round(decimals=3) + 0.
    array([[ 1.,  2.]])
    """

    # obtain the controllability matrix
    _, _, C = state_controllability(a, b)

    # obtain the observability matrix
    O = state_observability_matrix(a, c)

    # calculate the rank of the controllability and observability martix
    rank_C = numpy.linalg.matrix_rank(C)

    # transpose the observability matrix to calculate the column rank
    rank_O = numpy.linalg.matrix_rank(O.T)

    if rank_C <= rank_O:
        Ac, Bc, Cc = remove_uncontrollable_or_unobservable_states(a, b, c, C, rank=rank_C)

        O = state_observability_matrix(Ac, Cc)

        Aco, Bco, Cco = remove_uncontrollable_or_unobservable_states(Ac, Bc, Cc, O, uncontrollable=False, unobservable=True)

    else:
        Ao, Bo, Co = remove_uncontrollable_or_unobservable_states(a, b, c, O, uncontrollable=False, unobservable=True,
                                                                  rank=rank_O)

        _, _, C = state_controllability(Ao, Bo)

        Aco, Bco, Cco = remove_uncontrollable_or_unobservable_states(Ao, Bo, Co, C)

    return Aco, Bco, Cco

def minors(G, order):
    '''
    Returns the order minors of a MIMO tf G.
    '''
    minor = []
    Nrows, Ncols = G.shape
    for rowstokeep in itertools.combinations(range(Nrows), order):
        for colstokeep in itertools.combinations(range(Ncols), order):
            minor.append(G[rowstokeep,colstokeep].det().simplify())

    return minor


def lcm_of_all_minors(G):
    Nrows, Ncols = G.shape
    lcm = 1
    for i in range(1, min(Nrows, Ncols) + 1, 1):
        allminors = minors(G, i)
        for m in allminors:
            numer, denom = m.as_numer_denom()
            lcm = sympy.lcm(lcm, denom)
    return lcm


def poles(G):
    '''
    Return the poles of a multivariable transfer function system. Applies
    Theorem 4.4 (p135).

    Parameters
    ----------
    G : numpy matrix (n x n)
        The transfer function G(s) of the system.

    Returns
    -------
    zero : array
        List of zeros.

    Example
    -------
    >>> def G(s):
    ...     return 1 / (s + 2) * numpy.matrix([[s - 1,  4],
    ...                                       [4.5, 2 * (s - 1)]])
    >>> poles(G)
    [-2.00000000000000]

    Note
    ----
    Not applicable for a non-squared plant, yet.
    '''

    s = sympy.Symbol('s')
    G = sympy.Matrix(G(s))  # convert to sympy matrix object

    lcm = lcm_of_all_minors(G)

    pole = sympy.solve(lcm,s)

    return pole


def zeros(G=None, A=None, B=None, C=None, D=None):
    '''
    Return the zeros of a multivariable transfer function system for with
    transfer functions or state-space. For transfer functions, Theorem 4.5
    (p139) is used. For state-space, the method from Equations 4.66 and 4.67
    (p138) is applied.

    Parameters
    ----------
    G : numpy matrix (n x n)
        The transfer function G(s) of the system.
    A, B, C, D : numpy matrix
        State space parameters

    Returns
    -------
    zero : array
        List of zeros.

    Example
    -------
    >>> def G(s):
    ...     return 1 / (s + 2) * numpy.matrix([[s - 1,  4],
    ...                                        [4.5, 2 * (s - 1)]])
    >>> zeros(G)
    [4.00000000000000]

    Note
    ----
    Not applicable for a non-squared plant, yet. It is assumed that B,C,D will
    have values if A is defined.
    '''
    # TODO create a beter function to accept parameters and switch between tf and ss

    if G:
        s = sympy.Symbol('s')
        G = sympy.Matrix(G(s))  # convert to sympy matrix object

        lcm = lcm_of_all_minors(G)

        allminors = minors(G, G.rank())
        gcd = None
        for m in allminors:
            numer, denom = m.as_numer_denom()
            if denom != lcm:
                numer *= lcm/denom
            if numer.find('s'):
                if not gcd:
                    gcd = numer
                else:
                    gcd = sympy.gcd(gcd, numer)
        return sympy.solve(gcd, s)

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


def pole_zero_directions(G, vec, dir_type, display_type='a', e=0.00001):
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
    else: raise ValueError('Incorrect dir_type parameter')

    N = len(vec)
    if display_type == 'a':
        pz_dir = []
    else:
        pz_dir = numpy.matrix(numpy.zeros([G(e).shape[0], N]))
        valid = []

    for i in range(N):
        d = vec[i]
        g = G(d + e)

        U, _, V =  SVD(g)
        u = V[:, dt]
        y = U[:, dt]

# TODO complete validation test
        v = True

        if display_type == 'u':
            pz_dir[:, i] = u
            valid.append(v)
        elif display_type == 'y':
            pz_dir[:, i] = y
            valid.append(v)
        elif display_type == 'a':
            pz_dir.append((d, u, y, [v]))
        else: raise ValueError('Incorrect display_type parameter')

    if display_type == 'a':
        display = pz_dir
    else:
        display = pz_dir, valid

    return display


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

    yp_mat1 = numpy.matrix(numpy.diag(poles)) * \
                    numpy.matrix(numpy.ones([Np, Np]))
    yp_mat2 = yp_mat1.T
    Qp = (Yp.H * Yp) / (yp_mat1 + yp_mat2)

    yz_mat1 = (numpy.matrix(numpy.diag(zeros)) * \
              numpy.matrix(numpy.ones([Nz, Nz])))
    yz_mat2 = yz_mat1.T
    Qz = (Yz.H * Yz) / (yz_mat1 + yz_mat2)

    yzp_mat1 = numpy.matrix(numpy.diag(zeros)) * \
               numpy.matrix(numpy.ones([Nz, Np]))
    yzp_mat2 = numpy.matrix(numpy.ones([Nz, Np])) * \
               numpy.matrix(numpy.diag(poles))
    Qzp = Yz.H * Yp / (yzp_mat1 - yzp_mat2)

    if deadtime is None:

        pre_mat = sc_linalg.sqrtm((numpy.linalg.inv(Qz))).dot(Qzp).dot(sc_linalg.sqrtm(numpy.linalg.inv(Qp)))
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
            dead_time_matrix = numpy.diag(numpy.exp(numpy.multiply(dead_time_vec_max_row, s)))
            return dead_time_matrix

        Q_dead = numpy.zeros((Np,Np))

        for i in range(Np):
            for j in range(Np):
                numerator_mat = (numpy.transpose(numpy.conjugate(Yp[:, i])) *
                                   Dead_time_matrix(poles[i], dead_time_vec_max_row) * \
                                   Dead_time_matrix(poles[j], dead_time_vec_max_row) * Yp[:, j])
                denominator_mat = poles[i] + poles[j]
                Q_dead[i, j] = numerator_mat / denominator_mat

        lambda_mat = sc_linalg.sqrtm(numpy.linalg.pinv(Q_dead)) \
                        .dot(Qp + Qzp.dot(numpy.linalg.pinv(Qz)) \
                        .dot(numpy.transpose(numpy.conjugate(Qzp)))) \
                        .dot(sc_linalg.sqrtm(numpy.linalg.pinv(Q_dead)))

        # Final equation 6.19
        Ms_min = float(numpy.real(numpy.max(numpy.linalg.eig(lambda_mat)[0])))

    return Ms_min


def BoundKS(G, poles, up, e=0.00001):
    '''
    The functions uses equaption 6.24 (p229) to calculate the peak value for KS
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
    '''

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

    distCondNum : float
        The disturbance condition number :math:`\sigma` (G) :math:`\sigma` (G :math:`^{-1}` yd)

    yd : numpy matrix
        Disturbance direction.
    """

    gd1 = 1 / numpy.linalg.norm(gd, 2)  # Returns largest sing value of gd(wj)
    yd = gd1 * gd
    distCondNum = sigmas(G)[0] * sigmas(numpy.linalg.inv(G) * yd)[0]

    return gd1, numpy.array(yd).reshape(-1,).tolist(), distCondNum


def distRHPZ(G, Gd, RHP_Z):
    '''
    Applies equation 6.48 (p239) For performance requirements imposed by
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
    '''
    if numpy.real(RHP_Z) < 0: # RHP-z
        raise ValueError('Function only applicable to RHP-zeros')
    Yz, _ = pole_zero_directions(G, [RHP_Z], 'z', 'y')
    Dist_RHPZ = numpy.abs(Yz.H * Gd(RHP_Z))[0, 0]

    return Dist_RHPZ


# according to convention this procedure should stay at the bottom
if __name__ == '__main__':
    import doctest
    import sys

    # Exit with an error code equal to number of failed tests
    sys.exit(doctest.testmod()[0])
