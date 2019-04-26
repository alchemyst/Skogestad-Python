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
    def __init__(self, *system):
        N = len(system)
        if N == 1:  # is a utils.tf object
            if system[0] is not utils.tf:
                raise ValueError(f"InternalDelay expected an instance of utils.tf and got {type(system[0])}")

            lti = scipy.signal.lti(system[0].numerator, system[0].denominator).to_ss()
            delay = [system[0].deadtime]

            matrices = self.__lti_SS_to_InternalDelay_matrices(lti, delay)
            A, B1, B2, C1, C2, D11, D12, D21, D22, delays = matrices

        elif N == 2:  # is lti object with a delay term
            if system[0] is not scipy.signal.lti:
                raise ValueError(f"InternalDelay expected an instance of scipy.signal.lti and got {type(system[0])}")

            lti = system[0].to_ss()
            delay = system[1]

            matrices = self.__lti_SS_to_InternalDelay_matrices(lti, delay)
            A, B1, B2, C1, C2, D11, D12, D21, D22, delays = matrices

        elif N == 3: # assume that it is a num, den, delay
            if not numpy.all([sys is collections.Sequence for sys in system]):
                raise ValueError(f"InternalDelay expected numerator, denominator, delay arguments")

            lti = scipy.signal.lti(system[0], system[1]).to_ss()
            delay = system[2]

            matrices = self.__lti_SS_to_InternalDelay_matrices(lti, delay)
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
        if g2 is None:
            I_ss = scipy.signal.lti([1], [1]).to_ss()
            I_dt = [0]
            g2 = lti_SS_to_InternalDelay(I_ss, I_dt)

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
        D11_inv = numpy.linalg.inv(self.D11)

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


class TFDT:
    """
    A wrapper class for objects that behave like SISO transfer function objects.
    This class is meant to extend them so that they can handle time delays of the form:
    .. math::
        \sum_{i} G_i(s) e^{-D_is}

    where :math: `G_i(s)` is a rational function and :math: `D_i` is a positive real constant.
    """

    def __init__(self, tfs=None):
        if tfs is None:
            tfs = numpy.array([])
        elif isinstance(tfs, utils.tf):
            tfs = numpy.array([tfs])
        elif isinstance(tfs, collections.Sequence) and isinstance(tfs[0], utils.tf):
            tfs = numpy.array(tfs)
        else:
            raise TypeError("TFDT object can only be built out of utils.tf objects")

        self.tfs = tfs

    def step(self, *args, **kwargs):
        if "T" not in kwargs:
            T = numpy.linspace(0, 100, 1000)  # create default time space

        dt = T[1]
        ys_tot = numpy.zeros_like(T)

        for tf in self.tfs:
            _, ys = tf.step(*args, **kwargs)
            dts = int(numpy.round(tf.deadtime / dt))
            ys_tot += numpy.concatenate(numpy.zeros(dts), ys[dts:])

        return T, ys_tot

    def __repr__(self):
        reprs = [tf.__repr__() for tf in self.tfs]
        return reprs.join(" + ")

    def __call__(self, s):
        return sum([tf(s) for tf in self.tfs])

    def __add__(self, other):
        if isinstance(other, utils.tf):
            other = TFDT(other)

        if isinstance(other, TFDT):
            dt_dict = collections.defaultdict([])
            for tf in self.tfs:
                dt_dict[tf.deadtime].append(tf)

            for tf in other.tfs:
                dt_dict[tf.deadtime].append(tf)

            return TFDT([sum(group) for group in dt_dict.values()])
        else:
            raise TypeError(f"TFDT cannot add to an object of type {type(other)}")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, utils.tf):
            other = TFDT(other)

        ans = TFDT()
        if isinstance(other, TFDT):
            for tf_o in other.tfs:
                ans += TFDT([tf * tf_o for tf in self.tfs])
        return ans

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if not isinstance(other, tf):
            other = tf(other)
        return self * other.inverse()

    def __rtruediv__(self, other):
        return tf(other) / self

    def __div__(self, other):
        if not isinstance(other, tf):
            other = tf(other)
        return self * other.inverse()

    def __rdiv__(self, other):
        return tf(other) / self

    def __neg__(self):
        return tf(-self.numerator, self.denominator, self.deadtime)

    def __pow__(self, other):
        r = self
        for k in range(other - 1):
            r = r * self
        return r
