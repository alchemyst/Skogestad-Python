import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scs
from utils import Closed_loop

# Process model of G with various Controller Gains
# G(s) = 3*(-2*s+1)/((10*s+1)*(5*s+1))

"""Easy way around is to explicitly calculate the poles and zeros of the closed loop
or, to write  a program to do it for you """

def K_cl(KC):
    """just for proportional control
    the polynome in the numerator is just the Gain : Kz = [Kp]
    the polynome in the denominator for the controller is 1 Kp = [1]
    system's numerator Gz = [-6, 3]
    system's denominator Gp = [50, 15, 1]"""

    Kz = [KC]
    Kp = [1]
    Gz = [-6, 3]
    Gp = [50, 15, 1]

    # closed loop poles and zeros
    [Z_cl_poly, P_cl_poly] = Closed_loop(Kz, Kp, Gz, Gp)
    # calculating the response
    f = scs.lti(Z_cl_poly, P_cl_poly)
    tspan = np.linspace(0, 50, 100)
    [t, y] = f.step(0, tspan)
    plt.plot(t, y)


Kc = [0.5, 1.5, 2, 2.5]

#  calculating the time domian response
for K in Kc:
    K_cl(K)

plt.show()
