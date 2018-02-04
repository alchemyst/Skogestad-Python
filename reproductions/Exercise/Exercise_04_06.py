import sympy
import numpy
from utils import ssr_solve

a11, a12, a21, a22, b21, b22, z = sympy.symbols(
    'a11 a12 a21 a22 b21 b22 z'
)

A = sympy.Matrix([[a11, a12], [a21, a22]])
B = sympy.Matrix([[1, 1], [b21, b22]])
C = sympy.Matrix([[1, 0], [0, 1]])
D = sympy.Matrix(sympy.Matrix(numpy.zeros_like(A)))

print("Zeros = ", ssr_solve(A, B, C, D)['zeros'])

# Result confirmed by para 4.6.7.  C has full rank, and D = zero. No zero
# value s = z for which P loses rank