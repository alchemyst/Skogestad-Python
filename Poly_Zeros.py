import numpy
import scipy as sc
import scipy.signal as scs
import matplotlib.pyplot as plt


def Poly_Zeros_T(Poly_z_K, Poly_p_K, Poly_z_G, Poly_p_G):
    """Given the polynomial expansion in the denominator and numerator of the controller function K and G
    then this function return s the poles and zeros of the closed loop transfer function in terms of reference signal

    the arrays for the input must range from the highest order of the polynomial to the lowest"""

    Poly_z = numpy.polymul(Poly_z_K, Poly_z_G)
    Poly_p = numpy.polyadd(numpy.polymul(Poly_p_K, Poly_z_G), numpy.polymul(Poly_p_K, Poly_p_G))

    # return the poles and zeros of T
    Zeros = numpy.roots(Poly_z)
    Poles = numpy.roots(Poly_p)

    return Poles, Zeros


Kp = [1]
Kz = [1, 1]
Gp = [10, 1]
Gz = [1]

[P, Z] = Poly_Zeros_T(Kz, Kp, Gz, Gp)

print P, Z

scs.lti([1], [2, 4])
