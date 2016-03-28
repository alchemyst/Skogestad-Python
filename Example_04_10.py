from __future__ import print_function
import numpy as np
import sympy as sp
from utils import poles

def G(s):
    G = 1 / ((s + 1) * (s + 2) * (s - 1))
    return G * sp.Matrix([[(s - 1) * (s + 2), 0, (s - 1)**2],
                          [-(s + 1) * (s + 2), (s - 1) * (s + 1), (s - 1) * (s + 1)]])

s = sp.Symbol('s')
G = G(s)
print(G)

# TODO: This is a concept for a general m x n pole() function
R,C = np.shape(G)

m1 = []
print('---Minors of order 1---')
for r in range(0, R) :
    for c in range(0, C):
        if G[r, c] != 0:
            m1.append(G[r, c])
            print(G[r, c])

print('---Minors of order 2---')
m2_1 = G[:,[1,2]]
m2_2 = G[:,[0,2]]
m2_3 = G[:,[0,1]]
print(m2_1)
print(m2_2)
print(m2_3)

print('---All poles---')
m1count = np.shape(m1)[0]
for m in range(0, m1count):
    print(sp.solve(sp.denom(m1[m])))

def M21(s):
    return G[:,[1,2]]
def M12(s):
    return G[:,[0,1]]
def M13(s):
    return G[:,[0,1]]

print(poles(M21))
print(poles(M12))
print(poles(M13))

print('Therefore the poles are -1, 1 and -2')

##Usefull Sage example
#[x,y]=np.shape(G)
#elif x>y:
## change x to y if x<y
#    minors_of_all_orders = [G.minors(order) for order in range(1,x)]
#    non_zero_minors=[m for m in sum(minors_of_all_orders,[]) if m!=0]
#    CP=sum(non_zero_minors).full_simplify().denominator().factor()
#    ZP=sum(non_zero_minors).full_simplify().numerator().factor()
#    Poles=solve(CP==0,s)
#    P=[Poles[i].rhs() for i in range(len(Poles))]
#    Zeros=solve(ZP==0,s)
#    Z=[Zeros[i].rhs() for i in range(len(Zeros))]
