import utils
import numpy as np
import sympy as sp
from cmath import sqrt

s = utils.tf([1,0],[1])
L = (-s + 2)/(s*(s + 2))

_,_,_,w180 = utils.margins(L)
GM = 1/np.abs(L(1j*w180))
print('w_180',w180)
print('GM = 1/|L(w180)|',GM)
print('From 7.55, kmax = GM = ',GM)

omega = np.logspace(-2,2,1000)
s = 1j*omega

def rk(kmax):
    return (kmax - 1)/(kmax + 1)

def kbar(kmax):
    return (kmax + 1)/2

def RScondition(kmax):
    ineq = [abs(rk(kmax) * kbar(kmax) * L(i)/(1 + kbar(kmax)*L(i))) for i in s]
    return max(ineq)

# note: this itterative solution method was used, 
#because sympy's solve function does not accept equations with complex numbers

kmax = 1
ineq = 0
while ineq < 1:
    ineq = RScondition(kmax)
    kmax += 0.001

print('From 7.58, kmax = ',np.round(kmax)) 