import sympy as sp

from utils import pole, zero

s = sp.Symbol('s')
G = 1 / (1.25 * (s + 1) * (s + 2)) * sp.Matrix([[s - 1, s],
                                                [-6, s - 2]])
print 'Poles: ' , pole(G)
print 'Zeros: ' , zero(G)
