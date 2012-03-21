import numpy
import scipy
import scipy.signal
import matplotlib.pyplot as plt

w = numpy.logspace(-2, 1, 1000)
s = w*1j

for k in [0.1, 0.5, 1.0, 2.0]:
    L = k/s * (2 - s)/(2 + s)
#    w, L = scipy.signal.freqs([-k, 2*k],
#                              [1, 2, 0])
    T = L/(1+L)
    S = 1 - T
    plt.loglog(w, abs(S))
plt.show()

print s
