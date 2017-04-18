from utils import tf, RHPonly
import numpy as np


def add_deadtime_SISO(G, deadtime):
    G = (tf(list(G.numerator.coeffs),
            list(G.denominator.coeffs), deadtime=deadtime))
    return G

s = tf([1, 0], [1])
G = 5/((s - 3)*(10*s + 1))
Gd_noDT = 0.5/((s - 3)*(0.2*s + 1))
Gd = add_deadtime_SISO(Gd_noDT, 1.5)


def Gstable(G, poles):
    Gmul = 1
    for p in poles:
        Gmul *= (s - p)/(s + p)
    return Gmul*G


G_RHPpoles = RHPonly(G.poles())
Gs = Gstable(G, G_RHPpoles)
Gd_RHPpoles = RHPonly(Gd.poles())
Gds = Gstable(Gd_noDT, Gd_RHPpoles)

KSGd_bound = np.abs((Gs(G_RHPpoles[0]))**(-1) * Gds(G_RHPpoles[0]))
print('The lower bound on the peak of ||KSGd|| is: ', KSGd_bound)
