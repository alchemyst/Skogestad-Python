import utils
import numpy as np
import sympy as sp
from scipy.optimize import fsolve

s = utils.tf([1,0],[1])
L = (-s + 2)/(s*(s + 2))

_,_,_,w180 = utils.margins(L)
GM = 1/np.abs(L(1j*w180))
print('w_180',w180)
print('GM = 1/|L(w180)|',GM)
print('From 7.55, kmax = GM = ',GM)

omega = np.logspace(-2,2,1000)

def rk(kmax):
    return (kmax - 1)/(kmax + 1)

def kbar(kmax):
    return (kmax + 1)/2

def RScondition(kmax):
    abs_ineq = [abs(rk(kmax) * kbar(kmax) * L(1j*s)/(1 + kbar(kmax)*L(1j*s))) for s in omega]
    max_ineq = max(abs_ineq) - 1
    return max_ineq

kcal = fsolve(RScondition,1)
print('From 7.58, kmax = ',np.round(kcal,2))