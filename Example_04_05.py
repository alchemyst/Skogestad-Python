import sympy
import numpy as np

sympy.init_printing(use_latex='Mathjax')

t = sympy.Symbol('t')

A = sympy.Matrix([[-1/t, 0, 0, 0],
                  [1/t, -1/t, 0, 0],
                  [0, 1/t, -1/t, 0],
                  [0, 0, 1/t, -1/t]])

B = sympy.Matrix([[1/t],
                  [0],
                  [0],
                  [0]])

Q = sympy.Matrix([[1, -1, 1, -1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

# Input pole vector matrix
Up = B.H * Q
# Confirm by inspection that that pole vectors are non zero
print('Iput pole vector', Up)
# Consider linear combinations for repeated poles
print('No of uncontrollable states:', Q.rank() - Up.rank())

# Determine the rank of the controllability matrix
c_plus = [A**n*B for n in range(A.shape[1])]
control_matrix = np.hstack(c_plus)
# cannot take rank of control_matrix, therefore construct
# Sympy matrix
C_matrix = sympy.Matrix(np.zeros_like(A))
for r in range(0, A.shape[1]):
        for c in range(A.shape[1]):
            C_matrix[r,c] = control_matrix[r,c]
            
print('\nRank of controllability matix =', C_matrix.rank())                      
if C_matrix.rank() == C_matrix.shape[1] :
    print('Controllability', C_matrix, \
          '\nhas full rank and is state controllable')
else:
    print('Controllability', C_matrix, \
          '\ndoes not have full rank and is not state controllable')
