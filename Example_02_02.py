import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scs
from utils import feedback, tf

# Process model of G with various Controller Gains
s = tf([1, 0], 1)
G = 3*(-2*s + 1) / ((10*s + 1)*(5*s + 1))

tspan = np.linspace(0, 50, 100)

#  calculating the time domian response
for K in [0.5, 1.5, 2, 2.5]:
    T = feedback(G * K, 1)
    [t, y] = T.step(0, tspan)
    plt.plot(t, y)

plt.show()
