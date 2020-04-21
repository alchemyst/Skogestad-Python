import numpy as np
import scipy.linalg as spla

from .utils import state_controllability


def state_observability(A, C):
    """
    Parameters: A => state space matrix A
                C => state space matrix C

    Returns: state_obsr     => True is states are observable
             out_pole_vec   => The output vector poles y_p_i
             observe_matrix => The observability matrix
    """
    state_obsr = True

    A = np.asmatrix(A)
    C = np.asmatrix(C)

    # compute all output pole vectors
    ev, vr = spla.eig(A, left=False, right=True)
    out_pole_vec = [np.around(C.dot(x), 3) for x in vr.T]
    # TODO: is this calculation correct?
    state_obsr = not any(np.sum(x) == 0.0 for x in out_pole_vec)

    # compute observability matrix
    o_plus = [C*A**n for n in range(A.shape[1])]
    observe_matrix = np.vstack(o_plus)

    return state_obsr, out_pole_vec, observe_matrix


def is_min_realisation(A, B, C):
    """
    Parameters: A => state space matrix A
                B => state space matrix B
                C => state space matrix C

    Returns: is_min_real => True if the system is the minimum realisation
    """
    state_obsr, out_pole_vec, observe_matrix = state_observability(A, C)
    state_control, in_pole_vec, control_matrix = state_controllability(A, B)

    return state_control and state_obsr
