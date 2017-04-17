import sympy
import numpy as np

a11, a12, a21, a22, b21, b22, z = sympy.symbols('a11 a12 a21 a22 b21 b22 z')

A = sympy.Matrix([[a11, a12],[a21, a22]])
B = sympy.Matrix([[1, 1],[b21, b22]])
C = sympy.Matrix([[1, 0],[0, 1]])
D = sympy.Matrix(sympy.Matrix(np.zeros_like(A)))

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
# Result confirmed by para 4.6.7.  C has full rank, and D = zero.  No zero
# value s = z for which P loses rank