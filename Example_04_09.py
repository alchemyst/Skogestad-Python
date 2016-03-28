from __future__ import print_function
import sympy as sp

from utils import poles, zeros

def G(s):
    return 1 / (1.25 * (s + 1) * (s + 2)) * sp.Matrix([[s - 1, s],
                                                       [-6, s - 2]])

print('Poles: ' , poles(G))
print('Zeros: ' , zeros(G))
