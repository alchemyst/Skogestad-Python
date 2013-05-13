import numpy as np
import numpy.linalg as lin
import scipy.linalg as LA
import scipy.signal as sign

# This module executes Example 4.4.
# The controllability Gramian has been included

# Define state space matrices
A = np.matrix([[-2, -2],
               [0, -4]])
B = np.matrix([[1],
              [1]])
C = np.matrix([[1, 0]])
D = 0

# Create frequency domain mode;
G = sign.lti(A, B, C, D)

# Question: Why does A.transpose give the same eigenvectors as the book
# and not plain A??
# Answer: The eig function in the numpy library only calculates
# the right hand eigenvectors. The Scipy library provides a eig function
# in which you can specify whether you want left or right handed eigenvectors.
# To determine controllability you need to calculate the input pole vectors
# which is dependant on the left eigenvectors.

# Calculate eigen vectors and pole vectors
val, vec = LA.eig(A, None, 1, 0, 0, 0)
U1 = np.dot(B.T, vec[:, 0].T)
U2 = np.dot(B.T, vec[:, 1].T)

# Set-up controllability Matrix
Con = np.zeros([2, 2])
Con[:, 0] = B.T
Con[:, 1] = (A * B).T
n = lin.matrix_rank(Con)

P = LA.solve_lyapunov(A, -B * B.T)

# Display results
print '\nThe transfer function realization is:'
print 'G(s) = '
print np.poly1d(G.num[0], variable='s')
print "----------------"
print np.poly1d(G.den, variable='s')

print '\n1) Eigenvalues are: p1 = ', val[0], 'and p2 = ', val[1]
print '   with eigenvectors: q1 = ', vec[:, 0], 'and q2 = ', vec[:, 1]
print '   Input pole vectors are: up1 = ', U1, 'and up2 = ', U2

print '\n2) The controlabillity matrix has rank', n, 'and is given as:'
print '  ', Con[0, :], '\n  ', Con[1, :]

print '\n3) The controllability Gramian ='
print '  ', P[0, :], '\n  ', P[1, :]