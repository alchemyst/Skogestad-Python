'''
Created on 27 Mar 2013

@author: St Elmo Wilken
'''


import control as cn
import numpy as np
import scipy.linalg as spla

"""
This toolbox assumes you have the control toolbox at your
disposal... And all the imports above...
"""


def state_controllability(A, B):
    """
    Parameters: A => state space representation of matrix A
                B => state space representation of matrix B

    Returns: state_control (Bool) => True if state controllable
             in_pole_vec          => Input pole vectors for the states u_p_i
             control_matrix       => State Controllability Matrix

    This method checks of the state space description of the system
    is state controllable according to Skogestad section 4.2.

    Note: The Gramian matrix type of solution has already been implemented by
    the Control Toolbox folks.
    """
    state_control = True

    A = np.asmatrix(A)
    B = np.asmatrix(B)

    # computing all input pole vectors.
    ev, vl = spla.eig(A, left=True, right=False)
    in_pole_vec = [np.around(B.conj().T.dot(x)) for x in vl.T]
    sum_pole_vec = [np.sum(x) for x in in_pole_vec]
    num_zero = np.size(np.where(np.array(sum_pole_vec) == 0.0))
    if num_zero > 0:
        state_control = False

    # computing the controllability matrix
    c_plus = [A**n*B for n in range(A.shape[1])]
    control_matrix = np.hstack(c_plus)

    return state_control, in_pole_vec, control_matrix


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

    # computing all output pole vectors
    ev, vr = spla.eig(A, left=False, right=True)
    out_pole_vec = [np.around(C.dot(x), 3) for x in vr.T]
    sum_pole_vec = [np.sum(x) for x in out_pole_vec]
    num_zero = np.size(np.where(np.array(sum_pole_vec) == 0.0))
    if num_zero > 0:
        state_obsr = False

    # computing observability matrix
    o_plus = [C*A**n for n in range(A.shape[1])]
    observe_matrix = np.vstack(o_plus)

    return state_obsr, out_pole_vec, observe_matrix


def is_min_realisation(A, B, C):
    """
    Parameters: A => state space matrix
                B =>        ''
                C =>        ''

    Returns: is_min_real => True if the system is the minimum realisation
    """
    state_obsr, out_pole_vec, observe_matrix = state_observability(A, C)
    state_control, in_pole_vec, control_matrix = state_controllability(A, B)

    is_min_real = False

    if state_control == True and state_obsr == True:
        is_min_real = True

    return is_min_real


def zero_directions(zero_vec, tf, e=0.0001):
    """
    Parameters: zero_vec => a vector containing all the transmission zeros of a system
                tf       => the transfer function G(s) of the system
                e        => this avoids possible divide by zero errors in G(z)
    Returns:    zero_dir => zero directions in the form:
                            (zero, input direction, output direction)
    Notes: this method is going to give dubious answers if the function G has pole zero cancellation...
    """
    zero_dir = []
    for z in zero_vec:
        num, den = cn.tfdata(tf)
        rows, cols = np.shape(num)

        G = np.empty(shape=(rows, cols))

        for x in range(rows):
            for y in range(cols):
                top = np.polyval(num[x][y], z)
                bot = np.polyval(den[x][y], z)
                if bot == 0.0:
                    bot = e

                entry = float(top) / bot
                G[x][y] = entry

        U, S, V = np.linalg.svd(G)
        V = np.transpose(np.conjugate(V))
        u_rows, u_cols = np.shape(U)
        v_rows, v_cols = np.shape(V)
        yz = np.hsplit(U, u_cols)[-1]
        uz = np.hsplit(V, v_cols)[-1]
        zero_dir.append((z, uz, yz))
    return zero_dir
