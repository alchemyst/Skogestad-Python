# -*- coding: utf-8 -*-
"""
Created on Thu May 09 18:25:34 2013

@author: Simon Streicher
"""
from __future__ import print_function
import scipy.linalg as sc_lin
import numpy as np
import numpy.matlib

# Calculates the P, N and M matrices for the six different kinds
# of unstructured uncertainty in MIMO systems.

# Approach: Use arrays for easy manipulation and convert to matrices
# for linear algebra computations.

# TODO: Test accuracy with example as resiliency for handling
# arbitrary matrix dimensions unsure.


def G(s):
    """
    Give the transfer matrix of the system.
    """
    G = np.matrix([[1/(2*s + 1), 2],
                   [5/(s + 1), 3*(s + 1)/((s + 2)*(s + 4))]], dtype=complex)
    return G


def Wp(s):
    """
    Give the performance weight matrix.
    """
    Wp = np.matrix([[10, 10], [10, 10]])
    return Wp


def Wg(s):
    """
    Give the uncertainty weight matrix.
    This is the weight that is multiplied by the delta matrix and
    goes by different names depending on the form of unstructured uncertainty.
    """
    Wg = np.matrix([[10, 10], [10, 10]])
    return Wg


def Delta(s):
    """
    Give the delta complex perturbation marix.
    """
    Delta = np.matrix([[10]])
    return Delta


def K(s):
    """
    Give the transfer matrix of the controller.
    """
    K = np.matrix([[10, 10], [10, 10]])
    return K


def P(Ps):
    """
    Create concatenated P matrix from Ps matrix array
    """
    P = np.vstack([np.hstack([Ps[0, 0], Ps[0, 1], Ps[0, 2]]),
                   np.hstack([Ps[1, 0], Ps[1, 1], Ps[1, 2]]),
                   np.hstack([Ps[2, 0], Ps[2, 1], Ps[2, 2]])])
    return P


def partP(Ps):
    """
    Create partitions of P needed for lower LFT calculation from
    Ps matrix array
    """

    P11 = np.vstack([np.hstack([Ps[0, 0], Ps[0, 1]]),
                     np.hstack([Ps[1, 0], Ps[1, 1]])])
    P12 = np.vstack([Ps[0, 2], Ps[1, 2]])
    P21 = np.hstack([Ps[2, 0], Ps[2, 1]])
    P22 = Ps[2, 2]

    return P11, P12, P21, P22


# Specify the form of the unstructured uncertainty by allocating a value to
# variable FORM as follows:

# (Cross reference Figure 8.5 on page 293 of the 3rd edition of Skogestad)

#     1   -->     additive uncertainty
#     2   -->     multiplicative input uncertainty
#     3   -->     multiplicative output uncertainty
#     4   -->     inverse additive uncertainty
#     5   -->     inverse multiplicative input uncertainty
#     6   -->     inverse multiplicative output uncertainty

FORM = 1

# Specify the range and resolution of frequency response
omega = np.logspace(-3, 4, num=1000)


# Create a suitable identity and zero matrix
# TODO: Verify that correct dimension is used
dim = np.shape(G(1))[0]  # 1 (non-zero value) used to allow for ramps
I = np.matlib.identity(dim)
Z = np.matlib.zeros((dim, dim))

# Store the matrices in suitable indices for later
# partitioning and concatenation

Pstore = list()
Nstore = list()
Mstore = list()

for k in range(len(omega)):
    s = 1j * omega[k]

    if FORM == 1:
        # Define matrix P for additive uncertainty form
        
        Ps = np.array([[Z, Z,  Wg(s)],
                       [Wp(s), Wp(s), Wp(s)*G(s)],
                       [-I, -I, -G(s)]])

    if FORM == 2:
        # Define matrix P for multiplicative input uncertainty form
        
        Ps = np.array([[Z, Z,  Wg(s)],
                       [Wp(s)*G(s), Wp(s), Wp(s)*G(s)],
                       [-G(s), -I, -G(s)]])

    if FORM == 3:
        # Define matrix P for multiplicative output uncertainty form
        
        Ps = np.array([[Z, Z,  Wg(s)*G(s)],
                       [Wp(s), Wp(s), Wp(s)*G(s)],
                       [-I, -I, -G(s)]])

    if FORM == 4:
        # Define matrix P for inverse additive uncertainty form
        
        Ps = np.array([[G(s)*Wg(s), Z,  G(s)],
                       [Wp(s)*G(s)*Wg(s), Wp(s), Wp(s)*G(s)],
                       [-G(s)*Wg(s), -I, -G(s)]])

    if FORM == 5:
        # Define matrix P for inverse multiplicative uncertainty form
        
        Ps = np.array([[Wg(s), Z,  I],
                       [Wp(s)*G(s)*Wg(s), Wp(s), Wp(s)*G(s)],
                       [-G(s)*Wg(s), -I, -G(s)]])

    if FORM == 6:
        # Define matrix P for inverse multiplicative uncertainty form
        
        Ps = np.array([[Wg(s), Z,  G(s)],
                       [Wp(s)*Wg(s), Wp(s), Wp(s)*G(s)],
                       [-Wg(s), -I, -G(s)]])

    #  Calculate and store P matrix
    Pstore.append(P(Ps))

    Pp = partP(Ps)

    P11mat = np.mat(Pp[0])
    P12mat = np.mat(Pp[1])
    P21mat = np.mat(Pp[2])
    P22mat = np.mat(Pp[3])

    # Calculate N-matrix using lower LFT of P and K
    N = P11mat + P12mat * K(s) * sc_lin.inv((I - P22mat*K(s))) * P21mat
    Nstore.append(N)

    # Calculate M-matrix using the fact that it is
    # the first partition of the N-matrix.

    # TODO: Generalize means of handling dimensions
    M = N[0:dim][:, 0:dim]
    Mstore.append(M)

# Input the step to view
step = 500

print("The following results is for step: " + str(step))
print("and corresponds to a frequency of: " + str(omega[step]) + " rad/time")

print("The P-matrix is:")
print(Pstore[step])
print("")

print("The N-matrix is:")
print(Nstore[step])
print("")

print("The M-matrix is:")
print(Mstore[step])
