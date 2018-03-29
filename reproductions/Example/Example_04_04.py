from __future__ import print_function
import numpy as np
import numpy.linalg as lin
import scipy.linalg as LA
import scipy.signal as sign

from utils import state_controllability, zeros

# This module executes Example 4.4.
# The controllability Gramian has been included

# Define state space matrices
A = np.matrix([[-2, -2],
               [0, -4]])
B = np.matrix([[1],
              [1]])
C = np.matrix([[1, 0]])
D = np.matrix([[0]])

# Create frequency domain mode;
G = sign.lti(A, B, C, D)
control, in_vecs, c_matrix = state_controllability(A, B)

# Question: Why does A.transpose give the same eigenvectors as the book
# and not plain A??
# Answer: The eig function in the numpy library only calculates
# the right hand eigenvectors. The Scipy library provides a eig function
# in which you can specify whether you want left or right handed eigenvectors.
# To determine controllability you need to calculate the input pole vectors
# which is dependant on the left eigenvectors.

# Calculate eigen vectors and pole vectors
val, vec = LA.eig(A, None, 1, 0, 0, 0)

n = lin.matrix_rank(c_matrix)

P = LA.solve_continuous_lyapunov(A, -B * B.T)


# Display results
G_tf = G.to_tf()
print('\nThe transfer function realization is:')
print('G(s) = ')
print(np.poly1d(G_tf.num[0], variable='s'))
print("----------------")
print(np.poly1d(G_tf.den, variable='s'))

print('\n1) Eigenvalues are: p1 = ', val[0], 'and p2 = ', val[1])
print('   with eigenvectors: q1 = ', vec[:, 0], 'and q2 = ', vec[:, 1])
print('   Input pole vectors are: up1 = ', in_vecs[0], 'and up2 = ', in_vecs[1])

print('\n2) The controlabillity matrix has rank', n, 'and is given as:')
print(c_matrix)

print('\n3) The controllability Gramian =')
print('  ', P[0, :], '\n  ', P[1, :])

print('\nMore properties')
print("\nState Controllable: " + str(control))
print('Zeros: {0}'.format(zeros(None, A, B, C, D)))
