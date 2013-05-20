'''
Created on Jan 27, 2012

@author: Carl Sandrock
'''

import numpy
import scipy.signal


#import control
#tf = control.TransferFunction

def circle(cx, cy, r):
    npoints = 100
    theta = numpy.linspace(0, 2*numpy.pi, npoints)
    y = cx + numpy.sin(theta)*r
    x = cx + numpy.cos(theta)*r
    return x, y


def distance_from_nominal(w, k, tau, theta, nom_response):
    r = k/(tau*w*i + 1)*numpy.exp(-theta*w*i)
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


def findst(G, K):
    """ Find S and T given a value for G and K """
    L = G*K
    I = numpy.eye(G.outputs, G.inputs)
    S = inv(I + L)
    T = S*L
    return S, T


def phase(G, deg=False):
    return numpy.unwrap(numpy.angle(G, deg=deg), discont=180 if deg
    else numpy.pi)


def Closed_loop(Kz, Kp, Gz, Gp):
    """ Kz & Gz is the polynomial constants in the numerator
    Kp & Gp is the polynomial constants in the denominator """

    # calculating the product of the two polynomials in the numerator and denominator of transfer function GK
    Z_GK = numpy.polymul(Kz, Gz)
    P_GK = numpy.polymul(Kp, Gp)

    #calculating the polynomial of closed loop sensitivity function s = 1/(1+GK)
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
    """ Find the Greatest Common Divisor of two polynomials using Euclid's algorithm:
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
    """ Very basic transfer function object

    Construct with a numerator and denominator.

    >>> G = tf(1, [1, 1])
    >>> G
    tf([ 1.], [ 1.  1.])

    >>> G2 = tf(1, [2, 1])

    The object knows how to do addition:
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

    def __init__(self, numerator, denominator=1, deadtime=0):
        """ Initialize the transfer function from a numerator and denominator polynomial """
        self.numerator = numpy.poly1d(numerator)
        self.denominator = numpy.poly1d(denominator)
        self.simplify()
        self.deadtime = deadtime

    def inverse(self):
        """ inverse of the transfer function """
        return tf(self.denominator, self.numerator, -self.deadtime)

    def step(self, *args):
        return scipy.signal.lti(self.numerator, self.denominator).step(*args)

    def simplify(self):
        g = polygcd(self.numerator, self.denominator)
        self.numerator, remainder = self.numerator/g
        self.denominator, remainder = self.denominator/g

    def __repr__(self):
        r = "tf(" + str(self.numerator.coeffs) + ", " + str(self.denominator.coeffs)
        if self.deadtime != 0:
            r += ", deadtime=" + str(self.deadtime)
        r += ")"
        return r

    def __call__(self, s):
        """ This allows the transfer function to be evaluated at particular values of s
        Effectively, this makes a tf object behave just like a function of s

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


def sigmas(A):
    """ Return the singular values of A
    
    This is a convenience wrapper to enable easy calculation of singular values over frequency
    
    Example:
    >>> A = numpy.array([[1, 2], [3, 4]])
    >>> sigmas(A)
    array([ 5.4649857 ,  0.36596619])
    
    """
    #TODO: This should probably be created with functools.partial
    return numpy.linalg.svd(A, compute_uv=False)


def feedback(forward, backward):
    """ Calculate the feedback equivalent transfer function """
    #TODO: This should be made MIMO-aware
    return forward/(1 + forward*backward)


def sszeros(A, B, C, D, directions=False):
    """
    Calculate the zeros of a system in state space form

    Arguments:
      A, B, C, D : state space matrices
      directions : return direction vectors as well

    Outputs:
      z : a list containing the zeros
      xz, uz : arrays contiaining the corresponding zero directions 
               in the corresponding columns

    Note: This assumes a minimal realisation

    Example:
    >>> A = numpy.mat("[-2, -2; 0, -4]")
    >>> B = numpy.mat("[1; 1]")
    >>> C = numpy.mat("[1, 0]")
    >>> D = numpy.mat("0")
    >>> sszeros(A, B, C, D)
    array([-2.+0.j])

    >>> sszeros(A, B, C, D, directions=True)
    (array([-2.+0.j]), array([[ 0.0+0.j],
           [ 0.5+0.j]]), array([[ 1.+0.j]]))

    """
    A, B, C, D = [numpy.asmatrix(m) for m in [A, B, C, D]]

    # We proceed as in Skogestad, by solving
    #
    # (z Ig - M)v_z = 0, (with v_z = [x_z; u_z] ) (1)
    # first we calculate M and Ig
    M = numpy.bmat([[A, B], [C, D]])
    Ig = numpy.zeros_like(M)
    Ig[:A.shape[0], :A.shape[1]] = numpy.eye(*A.shape)

    #Now, scipy.linalg.eig solves
    # a   vr[:,i] = w[i]        b   vr[:,i] (2)
    # according to the docs.
    #
    # A little rearranging of (1) results in the form of (2):
    # z Ig v_z = M v_z
    # -> M v_z = z Ig v_z -> a=M, b=Ig, v_z=vr
    # so
    z, vz = scipy.linalg.eig(M, Ig)
    goodvalues = numpy.isfinite(z)
    if directions:
        xz, uz = numpy.split(vz, [A.shape[1]])
        return z[goodvalues], xz[:, goodvalues], uz[:, goodvalues]
    else:
        return z[goodvalues]

if __name__ == '__main__':
    import doctest
    doctest.testmod()
