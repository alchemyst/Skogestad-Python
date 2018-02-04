import sympy
import numpy
from utils import ssr_solve

c1, z = sympy.symbols('c1 z')

A = sympy.Matrix([[10, 0], [0, -1]])
B = sympy.Matrix([[1, 0], [0, 1]])
C = sympy.Matrix([[10, c1], [10, 0]])
D = sympy.Matrix([[0, 0], [0, 1]])

print("Zeros = ", ssr_solve(A, B, C, D)['zeros'])
# c1 must be less than 1 for RHP zero