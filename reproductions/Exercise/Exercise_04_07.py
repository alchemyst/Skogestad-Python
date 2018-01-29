import sympy
import numpy as np

c1, z = sympy.symbols('c1 z')

A = sympy.Matrix([[10, 0],[0, -1]])
B = sympy.Matrix([[1, 0],[0, 1]])
C = sympy.Matrix([[10, c1],[10, 0]])
D = sympy.Matrix([[0, 0],[0, 1]])

M = sympy.BlockMatrix([[A, B],
                       [C, D]])
Ig = np.zeros_like(M)
d = np.arange(A.shape[0])
Ig[d, d] = 1
Ig = sympy.Matrix(Ig)
zIg = z * Ig
P = sympy.Matrix(zIg - M)
zf = P.det()
print('solve z:', zf)
zero = sympy.solve(zf, z)
print("Zero = ",zero)
# c1 must be less than 1 for RHP zero