import numpy as np
import matplotlib.pyplot as plt
from utils import RGA

# The following code performs Example 3.12 of Skogestad.
# Here the RGA, iterative RGA, condition number and minimized
# condition number is calculated for constant transfer function G.
# The minimized coondition number is not implemented.
# Examples 3.13-15 are all similar

def condnum(A):
    gamma = A[0]/A[-1]
    return gamma

def IterRGA(A, n):
    for i in range(1, n):
        A = RGA(A)
    return A

def RGAnumber(A):
    RGAnum = np.sum(np.abs(RGA(A) - np.identity(len(A))))
    return RGAnum


G = np.matrix([[100, 0], [0, 1]])

[U, S, V] = np.linalg.svd(G)

R = RGA(G)
ItR = IterRGA(G, 4)
numR = RGAnumber(G)
numC = condnum(S)

print 'RGA:\n', R, '\nIterative RGA:\n', ItR, '\nCondition Number:\n', numC
