import numpy
import collections
import utils



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
