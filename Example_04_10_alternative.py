from __future__ import print_function
import sympy as sp
import utils

def G(s):
    return (1/((s+1)*(s+2)*(s-1)))*sp.Matrix([[(s-1)*(s+2), 0, (s-1)**2],[-(s+1)*(s+2), (s-1)*(s+1), (s-1)*(s+1)]])

poles = utils.poles(G)

print("The poles of the system are = ",poles)
