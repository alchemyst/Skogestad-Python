from utils import tf, RHPonly, polygcd
import numpy as np
import numpy.linalg as nplinalg
import sympy as sp
from sympy.core.compatibility import as_int

def add_deadtime_SISO(G, deadtime):
    G = (tf(list(G.numerator.coeffs), list(G.denominator.coeffs),deadtime = deadtime))
    return G

s = tf([1,0],[1])
G = 5/((s - 3)*(10*s + 1))
Gd_noDT = 0.5/((s - 3)*(0.2*s + 1))
Gd = add_deadtime_SISO(Gd_noDT,1.5)

def symbolic_poly(coeff):
    sym = 0
    s = sp.Symbol('s')
    for n in range(len(coeff)):
        sym = (sym + coeff[-n - 1] * s**n).simplify()
    return sym

def ceoff_symbolic_poly(expr):
    expr_poly = sp.Poly(expr)
    coeff  = [float(k) for k in expr_poly.all_coeffs()]
    return coeff

def Gstable(G,poles):
    Gmul = 1
    for p in poles:
        Gmul *= (s - p)/(s + p)
    Gs = Gmul*G
    num = Gs.numerator.c
    den = Gs.denominator.c
    sym_num = symbolic_poly(num)
    sym_den = symbolic_poly(den)
    fraction = (sym_num/sym_den).simplify()
    numer, denom = fraction.as_numer_denom()
    if numer.find('s'):
        num_coeff  = ceoff_symbolic_poly(numer)
    else:
        num_coeff = as_int(numer)
    if denom.find('s'):
        den_coeff  = ceoff_symbolic_poly(denom)
    else:
        den_coeff = denom
    return tf(num_coeff,den_coeff)


G_RHPpoles = RHPonly(G.poles())
Gs = Gstable(G,G_RHPpoles)
Gd_RHPpoles = RHPonly(Gd.poles())
Gds = Gstable(Gd_noDT,Gd_RHPpoles)

KSGd_bound = np.abs((Gs(G_RHPpoles[0]))**(-1) * Gds(G_RHPpoles[0]))
print('The lower bound on the peak of ||KSGd|| is: ', KSGd_bound)
