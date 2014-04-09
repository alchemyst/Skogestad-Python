#!/usr/bin/env python

import numpy
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
    """ Find S and T given a value for G and K """
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

    # the product of the two polynomials in the numerator
    # and denominator of transfer function GK
    Z_GK = numpy.polymul(Kz, Gz)
    P_GK = numpy.polymul(Kp, Gp)

    # the polynomial of closed loop
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
    Frequency in rad/time were time is the time unit used in model
    """
    omega = numpy.logspace(w_start, w_end, 1000)
    return omega


def freq(G):
    """ Calculate the frequency response for an optimisation problem

        Parameters
        ----------
        G :  plant transfer function

        Returns
        -------
        Gw : frequency response function

    """
    def Gw(w):
        return G(1j * w)
    return Gw


def ZeiglerNichols(G):
    """ Calculate the gain margin and phase margin of a system

        Parameters
        ----------

        G : plant transfer function

        Returns
        -------
        Kc : proportional gain
        Tauc : integral gain
        Ku : ultimate P controller gain
        Pu : corresponding period of oscillations

    """
    GM, PM, wc, w_180 = margins(G)
    Ku = numpy.abs(1 / G(1j * w_180))
    Pu = numpy.abs(2 * numpy.pi / w_180)
    Kc = Ku / 2.2
    Taui = Pu / 1.2

    return Kc, Taui, Ku, Pu


def margins(G):
    """ Calculate the gain margin and phase margin of a system

        Parameters
        ----------
        G : plant transfer function
        w1 : start frequency
        w2 : end frequency
        label : title for the figure (optional)
        margin : show the cross over frequencies on the plot (optional)

        Returns
        -------
        GM : gain margin
        PM : phase margin
        wc : gain crossover frequency
        w_180 : phase crossover frequency

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
    w_180 = optimize.fsolve(arg, -1)

    PM = numpy.angle(Gw(wc), deg=True) + 180
    GM = 1/(numpy.abs(Gw(w_180)))

    return GM, PM, wc, w_180


def marginsclosedloop(L):
    """ Calculate the margins of a closed loop system

        Parameters
        ----------
        L : loop transfer function

        Returns
        -------
        GM : gain margin
        PM : phase margin
        wc : gain crossover frequency for L
        wb : closed loop bandwidth for S
        wbt : closed loop bandwidth for T

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

    # calculate the freqeuncy at |S(jw)| = 0.707 from below
    wb = optimize.fsolve(modS, 0.1)
    # calculate the freqeuncy at |T(jw)| = 0.707 from above
    wbt = optimize.fsolve(modT, 0.1)

    #"Frequency range wb < wc < wbt
    if (PM < 90) and (wb < wc) and (wc < wbt):
        valid = True
    else: valid = False
    return GM, PM, wc, wb, wbt, valid


def bode(G, w1, w2, label='Figure', margin=False):
    """Give the Bode plot along with GM (gain margin) and PM (phase margin)

        Parameters
        ----------
        G : plant transfer function
        w1 : start frequency
        w2 : end frequency
        label : title for the figure (optional)
        margin : show the cross over frequencies on the plot (optional)

    """

    GM, PM, wc, w_180 = margins(G)

    # Bode plot with corresponding frequencies for PM and GM
#    if ((w2 < numpy.log(w_180)) and margin):
#        w2 = numpy.log(w_180)
    w = numpy.logspace(w1, w2, 1000)
    s = 1j*w

    plt.figure(label)
    plt.subplot(211)
    gains = numpy.abs(G(s))
    plt.loglog(w, gains)
    if margin:
        plt.loglog(wc*numpy.ones(2), [numpy.max(gains), numpy.min(gains)])
        plt.text(1, numpy.average([numpy.max(gains), numpy.min(gains)]), 'G(jw) = -180^o')
#        plt.loglog(w_180*numpy.ones(2), [numpy.max(gains), numpy.min(gains)])
    plt.loglog(w, 1 * numpy.ones(len(w)))
    plt.ylabel('Magnitude')

    # argument of G
    plt.subplot(212)
    phaseangle = phase(G(s), deg=True)
    plt.semilogx(w, phaseangle)
    if margin:
        plt.semilogx(wc*numpy.ones(2), [numpy.max(phaseangle), numpy.min(phaseangle)])
#        plt.semilogx(w_180*numpy.ones(2), [-180, 0])
    plt.ylabel('Phase')
    plt.xlabel('Frequency [rad/s]')

    return GM, PM

def bodeclosedloop(G, K, w1, w2, label='Figure', margin=False):
    """Give the Bode plot of the closed loop function (L, S, T)

        Parameters
        ----------
        G : plant transfer function
        K : controller transfer function
        w1 : start frequency
        w2 : end frequency
        label : title for the figure (optional)
        margin : show the cross over frequencies on the plot (optional)
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
    plt.ylabel("Magnitude")
    plt.legend(["L", "S", "T"],
               bbox_to_anchor=(0, 1.01, 1, 0), loc=3, ncol=3)

    if margin:
        plt.plot(w, 1/numpy.sqrt(2) * numpy.ones(len(w)), linestyle='dotted')

    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase(L, deg=True))
    plt.semilogx(w, phase(S, deg=True))
    plt.semilogx(w, phase(T, deg=True))
    plt.ylabel("Phase")
    plt.xlabel("Frequency [rad/s]")


# according to convention this procedure should stay at the bottom
if __name__ == '__main__':
    import doctest
    doctest.testmod()
