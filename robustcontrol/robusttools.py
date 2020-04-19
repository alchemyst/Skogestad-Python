import numpy as np
from . import utils


"""Use this file to add more MIMO functions for robust stability and performance"""


def wI(tau, ro, rinf, s):
    """
    Uncertainity weight function. Based on equation 7.38 (p273) and 8.28 (p297).

    Parameters
    ----------
    tau : float
        tau = 1 / x at y = 1.
    ro : float
        ro = y at x-low.
    rinf : float
        rinf = y at x-high.

    Returns
    -------
    wI : float
        Uncertainity weight.

    NOTE
    ----
    This is just one example of a uncertainity weight function.
    """

    return (tau * s + ro) / (tau / rinf * s + 1)


def UnstructuredDelta(M, DeltaStructure):
    """
    Function to calculate the unstructured delta stracture for a pertubation.

    Parameters
    ----------
    M : matrix (2 by 2),
        TODO: extend for n by n
        M-delta structure

    DeltaStructure : string ['Full','Diagonal']
        Type of delta structure

    Returns
    -------
    delta : matrix
        unstructured delta matrix
    """

    if DeltaStructure == "Full":
        [U, s, V] = utils.SVD(M)
        S = np.diag(s)
        delta = 1/s[0] * V[:,0] * U[:,0].H
    elif DeltaStructure == 'Diagonal':
# TODO: complete
        delta = 'NA'
    else: delta = 0
    return delta


def SpecRad(A):
    """
    Function to calculate the spectral radius, which is the magnitude of the
    largest eigenvalue of a matrix.

    Parameters
    ----------
    M : matrix (n by n)

    Returns
    -------
    rho : float
        Spectral radius.

    Note
    ----
    The spectral norm provides a lower bound for any matrix norm.
    """
    return max(np.linalg.eigvals(A))
