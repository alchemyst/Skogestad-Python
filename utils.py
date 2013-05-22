'''
Created on Jan 27, 2012

@author: Carl Sandrock
'''

import numpy
import scipy.signal
import matplotlib.pyplot as plt


#import control
#tf = control.TransferFunction

def circle(cx, cy, r):
    npoints = 100
    theta = numpy.linspace(0, 2*numpy.pi, npoints)
    y = cx + numpy.sin(theta)*r
    x = cx + numpy.cos(theta)*r
    return x, y


def distance_from_nominal(w, k, tau, theta, nom_response):
    r = k/(tau*w*1j + 1)*numpy.exp(-theta*w*1j)
    return numpy.abs(r - nom_response)


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
    """ Find S and T given a value for G and K """
    L = G*K
    I = numpy.eye(G.outputs, G.inputs)
    S = numpy.linalg.inv(I + L)
    T = S*L
    return S, T


def phase(G, deg=False):
    return numpy.unwrap(numpy.angle(G, deg=deg), discont=180 if deg
                        else numpy.pi)


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
    """ Calculate the Relative Gain Array of a matrix """
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
        
        return scipy.signal.lti(self.numerator, self.denominator).step(*args)

    def simplify(self):
        g = polygcd(self.numerator, self.denominator)
        self.numerator, remainder = self.numerator/g
        self.denominator, remainder = self.denominator/g
    
    def __repr__(self):
        if self.name != '':
            r = str(self.name) + "\n"
        else:
            r = ''
        r += "tf(" + str(self.numerator.coeffs) + ", " + str(self.denominator.coeffs)
        if self.deadtime != 0:
            r += ", deadtime=" + str(self.deadtime)
        r += ")"
        if self.u != '' and self.y != '':  
            r += "\ninput name: " + self.u
            r += "\noutput name: " + self.y
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

def tf_feedback(forward, backward='undefined', positive=False):
    """
    Defined for use in connect function
    Calculates a feedback loop
    This version is for trasnfer function objects
    Negative feedback is assumed, use positive=True for positive feedback
    Forward refers to the function that goes out of the comparator
    Backward refers to the function that goes into the comparator
    """

    # Create identity tf if no backward defined
    if backward == 'undefined':
        backward = tf(1)
    I = tf(1)
    if not positive:
        r = forward * tf.inverse((I + backward * forward))
    else:
        r = forward * tf.inverse((I - backward * forward))
    return r


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
    
    tspace = numpy.linspace(0, t_final, steps)
    foo = numpy.real(tf.step(initial_val, tspace))
    plt.plot(foo[0], foo[1])
    plt.show()

# TODO: Concatenate tf objects into MIMO structure


def sigmas(A):
    """
    Return the singular values of A

    This is a convenience wrapper to enable easy calculation of
    singular values over frequency

    Example:
    >> A = numpy.array([[1, 2],
                        [3, 4]])
    >> sigmas(A)
    array([ 5.4649857 ,  0.36596619])

    """
    #TODO: This should probably be created with functools.partial
    return numpy.linalg.svd(A, compute_uv=False)


def feedback(forward, backward='undefined', positive=False):
    """
    Calculates a feedback loop
    This version is for matrices
    Negative feedback is assumed, use positive=True for positive feedback
    Forward refers to the function that goes out of the comparator
    Backward refers to the function that goes into the comparator
    """

    # Create identity matrix if no backward matrix is specified
    if backward == 'undefined':
        backward = numpy.asmatrix(numpy.eye(numpy.shape(forward)[0],
                                  numpy.shape(forward)[1]))
    # Check the dimensions of the input matrices
    if numpy.shape(backward)[1] != numpy.shape(forward)[0]:
        raise ValueError("The column dimension of backward matrix must equal row dimension of forward matrix")
    forward = numpy.asmatrix(forward)
    backward = numpy.asmatrix(backward)
    I = numpy.asmatrix(numpy.eye(numpy.shape(backward)[0],
                                 numpy.shape(forward)[1]))
    if not positive:
        r = forward * numpy.linalg.inv((I + backward * forward))
    else:
        r = forward * numpy.linalg.inv((I - backward * forward))
    return r


if __name__ == '__main__':
    import doctest
    doctest.testmod()


def omega(w_start, w_end):
    """
    Convenience wrapper
    Defines the frequency range for calculation of frequency response
    Frequency in rad/time were time is the time unit used in model
    """
    omega = numpy.logspace(w_start, w_end, 1000)
    return omega