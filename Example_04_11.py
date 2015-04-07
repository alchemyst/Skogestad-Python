import sympy as sp

from utils import pole, zero

s = sp.Symbol('s')
G = 1 / (s + 2) * sp.Matrix([[s - 1,  4],
                             [4.5, 2 * (s - 1)]])
print 'Poles: ' , pole(G)
print 'Zeros: ' , zero(G)
