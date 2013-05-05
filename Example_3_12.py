import numpy as np
from utils import RGA

# The following code performs Example 3.12/3.13/3.14/3.15 of Skogestad.
# Here the RGA, iterative RGA and condition number is calculated for constant 
# transfer function G. 
# The minimized coondition number is not implemented.

def condnum(A):
    gamma = np.linalg.cond(A)
    return gamma

def IterRGA(A, n):
    for i in range(1, n):
        A = RGA(A)
    return A

def RGAnumber(A):
    RGAnum = np.sum(np.abs(RGA(A) - np.identity(len(A))))
    return RGAnum

def CalculateInverse(A):
    A_inv = np.linalg.inv(A)
    return A_inv
    
def CalculateSVD(A):
    [Umatrix,Sigma,Vmatrix] = np.linalg.svd(G)
    return [Umatrix,Sigma,Vmatrix]
 
# defines the transfer functions from each example   
G_3_12 = np.matrix([[100, 0], [0, 1]])
G_3_13 = np.matrix([[1,2], [0,1]])
G_3_14 = np.matrix([[87.8,-86.4], [108.2,-109.6]])
G_3_15 = np.matrix([[16.8, 30.5, 4.3], [-16.7, 31, -1.4], [1.27, 54.1, 5.4]])

# Calculates the SVD for each transfer function
[U1, S1, V1] = CalculateSVD(G_3_12)
[U2, S2, V2] = CalculateSVD(G_3_13)
[U3, S3, V3] = CalculateSVD(G_3_14)
[U4, S4, V4] = CalculateSVD(G_3_15)

# Calculates the inverse of each transfer function
G_3_12_inv = CalculateInverse(G_3_12)
G_3_13_inv = CalculateInverse(G_3_13)
G_3_14_inv = CalculateInverse(G_3_14)
G_3_15_inv = CalculateInverse(G_3_15)

# Calculate the RGA matrix for each example
R1 = RGA(G_3_12)
R2 = RGA(G_3_13)
R3 = RGA(G_3_14)
R4 = RGA(G_3_15)

# Calculates the interative RGA matrix for each example
ItR1 = IterRGA(G_3_12, 4)
ItR2 = IterRGA(G_3_13, 4)
ItR3 = IterRGA(G_3_14, 4)
ItR4 = IterRGA(G_3_15, 4)

# Calculates the RGA number for each example
numR1 = RGAnumber(G_3_12)
numR2 = RGAnumber(G_3_13)
numR3 = RGAnumber(G_3_14)
numR4 = RGAnumber(G_3_15)

# Calculates the condition number for each example
numC1 = condnum(G_3_12)
numC2 = condnum(G_3_13)
numC3 = condnum(G_3_13)
numC4 = condnum(G_3_15)

print '\nExample 3.12','\nTransfer Function:\n',G_3_12,'\nInverseG:\n',G_3_12_inv,'\nRGA:\n', R1, '\nIterative RGA:\n', ItR1, '\nCondition Number:\n', numC1
print '\nExample 3.13','\nTransfer Function:\n',G_3_13,'\nInverseG:\n',G_3_13_inv,'\nRGA:\n', R2, '\nIterative RGA:\n', ItR2, '\nCondition Number:\n', numC2
print '\nExample 3.14','\nTransfer Function:\n',G_3_14,'\nInverseG:\n',G_3_14_inv,'\nRGA:\n', R3, '\nIterative RGA:\n', ItR3, '\nCondition Number:\n', numC3
print '\nExample 3.15','\nTransfer Function:\n',G_3_15,'\nInverseG:\n',G_3_15_inv,'\nRGA:\n', R4, '\nIterative RGA:\n', ItR4, '\nCondition Number:\n', numC4
