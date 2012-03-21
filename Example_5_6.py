import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scs

# Process model of G with various Controller Gains
# G(s) = (-s+1)/(s+1)
#controller k1 = kc*((s+1)/s)*(1/(0.05*s+1))

# explicitly calculate the poles and zeros of the closed loop transfer function


def Closed_loop(Kz, Kp, Gz, Gp):
    """Kz & Gz is the polynomial constants in the numerator
    Kp & Gp is the polynomial constants in the denominator"""

    # calculating the product of the two polynomials in the numerator and denominator of transfer function GK
    Z_GK = np.polymul(Kz, Gz)
    P_GK = np.polymul(Kp, Gp)

    #calculating the polynomial of closed loop sensitivity function s = 1/(1+GK)
    Zeros_poly = Z_GK
    Poles_poly = np.polyadd(Z_GK, P_GK)
    return Zeros_poly, Poles_poly


def K_cl(KC):
    """system's numerator Gz = [-1, 1]
    system's denominator Gp = [1, 1]"""

    Kz = [KC, KC]
    Kp = [0.05, 1, 0]
    Gz = [-1, 1]
    Gp = [1, 1]

    # closed loop poles and zeros
    [Z_cl_poly, P_cl_poly] = Closed_loop(Kz, Kp, Gz, Gp)
    # calculating the response
    f = scs.lti(Z_cl_poly, P_cl_poly)
    tspan = np.linspace(0, 5, 100)
    [t, y] = f.step(0, tspan)
    plt.subplot(2, 1, 1)
    plt.plot(t, y)
    plt.xlabel('time (sec)')
    plt.ylabel('y(t)')


Kc = [0.2, 0.5, 0.8]

#  calculating the time domian response
for K in Kc:
    K_cl(K)

#sensitivity function
w = np.logspace(-2, 2, 1000)
s = w*1j
for kc in [0.2, 0.5, 0.8]:
    G = (-s+1)/(s+1)
    #RHP_zero at s = 1
    k1 = kc*((s+1)/s)*(1/(0.05*s+1))
    L = k1*G
    T = L/(1+L)
    S = 1-T
    plt.subplot(2, 1, 2)
    plt.loglog(w, abs(S))

plt.xlabel('frequency (rad/s)')
plt.ylabel('magnitude (S)')
plt.show()
