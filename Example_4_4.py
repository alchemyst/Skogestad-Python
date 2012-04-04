import numpy as np
import numpy.linalg as lin
import scipy as sc
import scipy.signal as sign

# This module executes Example 4.4, but doesn't include
# the controllability Gramian.

# Define state space matrices
A = np.matrix([[-2, -2], 
               [0, -4]])
B = np.array([[1], 
              [1]])
C = np.array([[1, 0]])
D = 0

# Create frequency domain mode;
G = sign.lti(A, B, C, D)

# Question: Why does A.transpose give the same eigenvectors as the book and not plain A??
# Calculate eigen vectors and pole vectors
val, vec = lin.eig(A.T)
U = B.T*vec

# Set-up controlability Matrix
Con = np.zeros([2, 2])
Con[:, 0] = B.T
Con[:, 1] = (A*B).T
n = lin.matrix_rank(Con)

# Display results
print '\nThe Transfer function realization is:'
print 'G(s) = ', '(', G.num[0, 1], 's +', G.num[0, 2], ') / (', G.den[0], 's^2 +', G.den[1], 's +', G.den[2], ')'

print '\n1) Eigenvalues are: p1 = ', val[1], 'and p2 = ', val[1]
print '   With eigenvectors: q1 = ', vec[:, 1].T, 'and q2 = ', vec[:, 0].T
print '   Output pole vectors are: up1 = ', U[0, 1], 'and up2 = ', U[0, 0]

print '\n2) The controlabillity matrix has rank', n, 'and is give as:'
print '  ', Con[0, :], '\n  ', Con[1, :]

print '\n3) The controllability Gramian can be determined by using \n   control.matlab.lyap which utilizes the control and Slicot library'
