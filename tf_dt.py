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

    def __init__(self, tfs):
        if isinstance(tfs, utils.tf):
            tfs = [tfs]
        elif isinstance(tfs, collections.Sequence) and isinstance(tfs[0], utils.tf):
            tfs = numpy.array(tfs)
        else:
            raise TypeError("TFDT object can only be built out of utils.tf objects")

        self.tfs = tfs

    def step(self, *args, **kwargs):
        pass


import matplotlib.pyplot as plt
a = utils.tf([1], [1, 1])
ts, ys = a.step(T=numpy.linspace(0, 100, 1000))

plt.plot(ts, ys)
plt.show()
