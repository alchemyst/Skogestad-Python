# -*- coding: utf-8 -*-
"""
Created on Apr 19, 2019

@author: Darren Roos (http://github.com/darren-roos/)
"""

import numpy
import collections
import utils
import scipy.signal


class InternalDelay:
    r"""
    A class for systems that have internal delays that can be represented as
    .. math::
        \dot{x} = A x + B_{1} u + B_{2}  w
        y = C_{1} x + D_{11} u + D_{12} w
        z = C_{2} x + D_{21} u + D_{22} w
        w_i = z_i(t - delay_i)

    Parameters
    ----------
    *system: arguments
        The `InternalDelay` class can be instantiated with 1, 2, 3 or 10
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1:    `utils.tf` system
            * 2:    `scipy.signal.lti`: system
                    array-like:  delays
            * 3:    array-like: (numerator, denominator, delays)
            * 10:   2-dimensional array-like: A, B1, B2, C1, C2, D11, D12, D21, D22
                    array-like: delays

    Examples
    --------
    Construct the example with feedforward control found here:
    https://www.mathworks.com/help/control/examples/specifying-time-delays.html#d120e709
    >>> P_id = InternalDelay([5], [1, 1], [3.4])
    >>> C_id = utils.InternalDelay([0.1*5, 0.1], [5, 0], [0])
    >>> cas_id = C_id * P_id
    >>> fed_id = cas_id.feedback()
    >>> F_id = utils.InternalDelay([0.3], [1, 4], [0])
    >>> I_id = utils.InternalDelay([1], [1], [0])
    >>> GKI_id = (P_id * C_id + I_id)**(-1)
    >>> PF_id = P_id * F_id
    >>> PFGKI_id = PF_id * GKI_id
    >>> TF_id = PFGKI_id + fed_id

    Simulate the example
    >>> uf = lambda t: numpy.array([1])
    >>> ts = numpy.linspace(0, 100, 1000)
    >>> ys = par_id.simulate(uf, ts)
    """

    def __init__(self, *system):
        N = len(system)
        if N == 1:  # is a utils.tf object
            if not isinstance(system[0], utils.tf):
                raise ValueError(f"InternalDelay expected an instance of utils.tf and got {type(system[0])}")

            lti = scipy.signal.lti(system[0].numerator, system[0].denominator).to_ss()
            delay = [system[0].deadtime]

            matrices = InternalDelay.__lti_SS_to_InternalDelay_matrices(lti, delay)
            A, B1, B2, C1, C2, D11, D12, D21, D22, delays = matrices

        elif N == 2:  # is lti object with a delay term
            if not isinstance(system[0], scipy.signal.lti):
                raise ValueError(f"InternalDelay expected an instance of scipy.signal.lti and got {type(system[0])}")

            lti = system[0].to_ss()
            delay = system[1]

            matrices = InternalDelay.__lti_SS_to_InternalDelay_matrices(lti, delay)
            A, B1, B2, C1, C2, D11, D12, D21, D22, delays = matrices

        elif N == 3: # assume that it is a num, den, delay
            if not numpy.all([isinstance(sys, collections.Sequence) for sys in system]):
                raise ValueError(f"InternalDelay expected numerator, denominator, delay arguments")

            lti = scipy.signal.lti(system[0], system[1]).to_ss()
            delay = system[2]

            matrices = InternalDelay.__lti_SS_to_InternalDelay_matrices(lti, delay)
            A, B1, B2, C1, C2, D11, D12, D21, D22, delays = matrices

        elif N == 10:
            A, B1, B2, C1, C2, D11, D12, D21, D22, delays = system

        else:
            raise ValueError("InternalDelay cannot be constructed out of input given")

        self.A = A
        self.B1 = B1
        self.B2 = B2
        self.C1 = C1
        self.C2 = C2
        self.D11 = D11
        self.D12 = D12
        self.D21 = D21
        self.D22 = D22
        self.delays = numpy.array(delays)

    def __lti_SS_to_InternalDelay_matrices(P_ss, P_dt):
        """
        Converts a SISO `scipy.signal.lti` object into the correct matrices
        for an internal delay calculations
        """
        A = P_ss.A
        C1 = P_ss.C
        C2 = numpy.zeros_like(P_ss.C)

        D22 = numpy.zeros_like(P_ss.D)

        if P_dt == [0]:
            B1 = P_ss.B
            B2 = numpy.zeros_like(P_ss.B)

            D11 = P_ss.D
            D12 = numpy.zeros_like(P_ss.D)
            D21 = numpy.zeros((P_ss.D.shape[0], P_ss.D.shape[0]))


        else:
            B1 = numpy.zeros_like(P_ss.B)
            B2 = P_ss.B

            D11 = numpy.zeros_like(P_ss.D)
            D12 = P_ss.D
            D21 = numpy.eye(P_ss.D.shape[0])

        delays = P_dt

        return A, B1, B2, C1, C2, D11, D12, D21, D22, delays

    def cascade(self, g2):
        """
        Calculates the `InternalDelay` object formed when combining two
        `InternalDelay` objects in series in the following order:

        -----> G2 -----> G1 ------->

        where G1 is `self`.
        """
        A = numpy.block([[self.A, numpy.zeros((self.A.shape[0], g2.A.shape[1]))],
                         [g2.B1 @ self.C1, g2.A]])

        B1 = numpy.block([[self.B1],
                          [g2.B1 @ self.D11]])

        B2 = numpy.block([[self.B2, numpy.zeros((self.B2.shape[0], g2.B2.shape[1]))],
                          [g2.B1 @ self.D12, g2.B2]])

        C1 = numpy.block([g2.D11 @ self.C1, g2.C1])

        C2 = numpy.block([[self.C2, numpy.zeros((self.C2.shape[0], g2.C2.shape[1]))],
                          [g2.D21 @ self.C1, g2.C2]])

        D11 = g2.D11 @ self.D11

        D12 = numpy.block([g2.D11 @ self.D12, g2.D12])

        D21 = numpy.block([[self.D21],
                           [g2.D21 @ self.D11]])

        D22 = numpy.block([[self.D22, numpy.zeros((self.D22.shape[0], g2.D22.shape[1]))],
                           [g2.D21 @ self.D12, g2.D22]])

        delays = numpy.block([self.delays, g2.delays])

        return InternalDelay(A, B1, B2, C1, C2, D11, D12, D21, D22, delays)

    def feedback(self, g2=None):
        """
        Calculates the `InternalDelay` object formed when combining two
        `InternalDelay` objects in a feedback loop in the following manner:

        ------>+ o -----> G1 ------->
                 ^-             |
                 |              |
                 |              |
                 |<---- G2 <----|

        where G1 is `self`, and if G2 is not given, it is assumed that
        G2 is an identity matrix.
        """
        if g2 is None:
            g2 = InternalDelay([1], [1], [0])

        X_inv = numpy.linalg.inv(numpy.eye(g2.D11.shape[0]) + g2.D11 @ self.D11)

        A = numpy.block([
            [self.A - self.B1 @ X_inv @ g2.D11 @ self.C1,
             -self.B1 @ X_inv @ g2.C1],
            [g2.B1 @ self.C1 - g2.B1 @ self.D11 @ X_inv @ g2.D11 @ self.C1,
             g2.A - g2.B1 @ self.D11 @ X_inv @ g2.C1]])

        B1 = numpy.block([[self.B1 - self.B1 @ X_inv @ g2.D11 @ self.D11],
                          [g2.B1 @ self.D11 - g2.B1 @ self.D11 @ X_inv @ g2.D11 @ self.D11]])

        B2 = numpy.block([
            [self.B2 - self.B1 @ X_inv @ g2.D11 @ self.D12,
             -self.B1 @ X_inv @ g2.D12],
            [g2.B1 @ self.D12 - g2.B1 @ self.D11 @ X_inv @ g2.D11 @ self.D12,
             g2.B2 - g2.B1 @ self.D11 @ X_inv @ g2.D12]])

        C1 = numpy.block([self.C1 - self.D11 @ X_inv @ g2.D11 @ self.C1,
                          -self.D11 @ X_inv @ g2.C1])

        C2 = numpy.block([
            [self.C2 - self.D21 @ X_inv @ g2.D11 @ self.C1,
             -self.D21 @ X_inv @ g2.C1],
            [g2.D21 @ self.C1 - g2.D21 @ self.D11 @ X_inv @ g2.D11 @ self.C1,
             g2.C2 - g2.D21 @ self.D11 @ X_inv @ g2.C1]])

        D11 = self.D11 - self.D11 @ X_inv @ g2.D11 @ self.D11

        D12 = numpy.block([self.D12 - self.D11 @ X_inv @ g2.D11 @ self.D12,
                           - self.D11 @ X_inv @ g2.D12])

        D21 = numpy.block([[self.D21 - self.D21 @ X_inv @ g2.D11 @ self.D11],
                           [g2.D21 @ self.D11 - g2.D21 @ self.D11 @ X_inv @ g2.D11 @ self.D11]])

        D22 = numpy.block([
            [self.D22 - self.D21 @ X_inv @ g2.D11 @ self.D12,
             -self.D21 @ X_inv @ g2.D12],
            [g2.D21 @ self.D12 - g2.D21 @ self.D11 @ X_inv @ g2.D11 @ self.D12,
             g2.D22 - g2.D21 @ self.D11 @ X_inv @ g2.D12]])

        delays = numpy.block([self.delays, g2.delays])

        return InternalDelay(A, B1, B2, C1, C2, D11, D12, D21, D22, delays)

    def parallel(self, g2):
        """
        Calculates the `InternalDelay` object formed when combining two
        `InternalDelay` objects in parallel in the following manner:

        -----> G1 ------->|
                          |
                          v+
                          o ---->
                          ^+
                          |
        -----> G2 ------->|

        where G1 is `self`.
        """

        A = numpy.block([[self.A, numpy.zeros((self.A.shape[0], g2.A.shape[1]))],
                         [numpy.zeros((g2.A.shape[0], self.A.shape[1])), g2.A]])

        B1 = numpy.block([[self.B1],
                          [g2.B1]])

        B2 = numpy.block([[self.B2, numpy.zeros((self.B2.shape[0], g2.B2.shape[1]))],
                          [numpy.zeros((g2.B2.shape[0], self.B2.shape[1])), g2.B2]])

        C1 = numpy.block([self.C1, g2.C1])

        C2 = numpy.block([[self.C2, numpy.zeros((self.C2.shape[0], g2.C2.shape[1]))],
                          [numpy.zeros((g2.C2.shape[0], self.C2.shape[1])), g2.C2]])

        D11 = self.D11 + g2.D11

        D12 = numpy.block([self.D12, g2.D12])

        D21 = numpy.block([[self.D21],
                           [g2.D21]])

        D22 = numpy.block([[self.D22, numpy.zeros((self.D22.shape[0], g2.D22.shape[1]))],
                           [numpy.zeros((g2.D22.shape[0], self.D22.shape[1])), g2.D22]])

        delays = numpy.block([self.delays, g2.delays])

        return InternalDelay(A, B1, B2, C1, C2, D11, D12, D21, D22, delays)

    def inverse(self):
        """
        Calculates the `InternalDelay` object formed when inverting an
        `InternalDelay` object.
        """
        try:
            D11_inv = numpy.linalg.inv(self.D11)
        except numpy.linalg.LinAlgError:
            raise numpy.linalg.LinAlgError("Cannot invert InternalDelay object: inverse is not physically realisable")

        A = self.A - self.B1 @ D11_inv @ self.C1

        B1 = self.B1 @ D11_inv

        B2 = self.B2 - self.B1 @ D11_inv @ self.D12

        C1 = -D11_inv @ self.C1

        C2 = self.C2 - self.D21 @ D11_inv @ self.C1

        D11 = D11_inv

        D12 = - D11_inv @ self.D12

        D21 = self.D21 @ D11_inv

        D22 = self.D22 - self.D21 @ D11_inv @ self.D12

        delays = self.delays

        return InternalDelay(A, B1, B2, C1, C2, D11, D12, D21, D22, delays)

    def simulate(self, uf, ts, x0=None):
        """
        Simulates the response of the system to the input.
        Uses a Runge-Kutta delay integration routine.
        Parameters:
            uf:     a callable object with the calling signature uf(t), where t is a scalar.
                    Defines the input to the system.
                    Should return a list-like object with an element for each input.
            ts:     a list-like objects of times over which the integration should be done.
                    Number of point should be at least 10 times more than the span
            x0:     Initial conditions for state of the system. Defaults to zero

        Returns:
            ys:     A list-like object containing the response of the system.
                    The shape of the object is (T, O), where T is the number of time
                    steps and O is the number of outputs.

        """
        if x0 is None:
            x0 = numpy.zeros(self.A.shape[0])

        dt = ts[1]
        dtss = [int(numpy.round(delay / dt)) for delay in self.delays]
        zs = []

        def wf(t):
            ws = []
            for i, dts in enumerate(dtss):
                if len(zs) <= dts:
                    ws.append(0)
                elif dts == 0:
                    ws.append(zs[-1][i])
                else:
                    ws.append(zs[-dts][i])

            return numpy.array(ws)

        f = lambda t, x: self.A @ x + self.B1 @ uf(t) + self.B2 @ wf(t)

        xs = [x0]
        ys = []
        for t in ts:
            x = xs[-1]

            # y
            y = self.C1 @ numpy.array(x) + self.D11 @ uf(t) + self.D12 @ wf(t)
            ys.append(list(y))

            # z
            z = self.C2 @ numpy.array(x) + self.D21 @ uf(t) + self.D22 @ wf(t)
            zs.append(list(z))

            # x integration
            k1 = f(t, x) * dt
            k2 = f(t + 0.5 * dt, x + 0.5 * k1) * dt
            k3 = f(t + 0.5 * dt, x + 0.5 * k2) * dt
            k4 = f(t + dt, x + k3) * dt
            dx = (k1 + k2 + k2 + k3 + k3 + k4) / 6
            x = [xi + dxi for xi, dxi in zip(x, dx)]
            xs.append(list(x))

        return numpy.array(ys)

    def __add__(self, other):
        return self.parallel(other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.cascade(other)

    def __rmul__(self, other):
        return other.cascade(self)

    def __truediv__(self, other):
        return self * other.inverse()

    def __rtruediv__(self, other):
        return other * self.inverse()

    def __div__(self, other):
        return self * other.inverse()

    def __rdiv__(self, other):
        return other * self.inverse()

    def __neg__(self):
        matrices = self.A, self.B1, self.B2, self.C1, self.C2, self.D11, self.D12, self.D21, self.D22, self.delays
        A, B1, B2, C1, C2, D11, D12, D21, D22, delays = matrices

        return InternalDelay(A, -B1, B2, C1, C2, -D11, D12, -D21, D22, delays)

    def __pow__(self, power):
        if not isinstance(power, int):
            raise ValueError("Cannot raise object to non-integer power")

        if power == 0:
            return InternalDelay([1], [1], [0])

        r = self
        if power < 0:
            r = self.inverse()

        for k in range(power-1):
            r = r * self

        return r
