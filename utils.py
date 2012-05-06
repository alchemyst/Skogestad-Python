'''
Created on Jan 27, 2012

@author: Carl Sandrock
'''

import numpy

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
    """ recurses down to scalar elements in A, then applies f, returning lists containing the result"""
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
    return numpy.unwrap(numpy.angle(G, deg=deg), discont=180 if deg else numpy.pi)


def Closed_loop(Kz, Kp, Gz, Gp):
    """Kz & Gz is the polynomial constants in the numerator
    Kp & Gp is the polynomial constants in the denominator"""

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


class tf(object):
    """ Very basic transfer function object 
    
    Construct with a numerator and denominator.  

    >>> G = tf(1, [1, 1])
    >>> G
    tf([1], [1 1])
    
    >>> G2 = tf(1, [2, 1])
    
    The object knows how to do addition:
    >>> G + G2
    tf([3 2], [2 3 1])
    
    multiplication
    >>> G * G2
    tf([1], [2 3 1])
    
    division
    >>> G / G2
    tf([2 1], [1 1])
    
    Deadtime is supported:
    >>> G3 = tf(1, [1, 1], deadtime=2)
    >>> G3
    tf([1], [1 1], deadtime=2)
    
    Note we can't add transfer functions with different deadtime:
    >>> G2 + G3
    Traceback (most recent call last):
        ...
    ValueError: Transfer functions can only be added if their deadtimes are the same
    
    """

    def __init__(self, numerator, denominator=1, deadtime=0):
        """ Initialize the transfer function from a numerator and denominator polynomial """
        self.numerator = numpy.poly1d(numerator)
        self.denominator = numpy.poly1d(denominator)
        self.deadtime = deadtime

    def inverse(self):
        """ inverse of the transfer function """
        return tf(self.denominator, self.numerator, -self.deadtime)

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
        if self.deadtime != other.deadtime:
            raise ValueError("Transfer functions can only be added if their deadtimes are the same")
        gcd = self.denominator * other.denominator
        return tf(self.numerator*other.denominator + 
                  other.numerator*self.denominator, gcd, self.deadtime)

    def __mul__(self, other):
        return tf(self.numerator*other.numerator,
                  self.denominator*other.denominator,
                  self.deadtime + other.deadtime)

    def __div__(self, other):
        return self * other.inverse()

if __name__ == '__main__':
    import doctest
    doctest.testmod()