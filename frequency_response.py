import numpy
import matplotlib.pyplot as plt

def unwrapped_angle(Gfr):
    return numpy.unwrap(numpy.arctan2(numpy.imag(Gfr), numpy.real(Gfr)))

def bode(w, Gfr):
    plt.subplot(2, 1, 1)
    plt.loglog(w, numpy.abs(Gfr))
    plt.subplot(2, 1, 2)
    plt.semilogx(w, unwrapped_angle(Gfr))

w = numpy.logspace(-2, 2)
s = w * 1j

for power in range(1, 4):
    Gfr = 1/(s + 1)**power
    bode(w, Gfr)

plt.show()