import numpy as np
import scipy.linalg as spla
import sympy as sp


def state_controllability(A, B):
    """
    Parameters: A => matrix A of state space representation
                B => matrix B of state space representation 

    Returns: state_control (Bool) => True if state controllable
             in_pole_vec          => Input pole vectors for the states u_p_i
             control_matrix       => State Controllability Matrix

    This method checks if the state space description of the system
    is state controllable according to Skogestad section 4.2.
    
    Note: This does not check for state controllability for systems with 
    repeated poles

    Note: The Gramian matrix type of solution has already been implemented by
    the Control Toolbox folks.
    """
    state_control = True

    A = np.asmatrix(A)
    B = np.asmatrix(B)
        
    # Compute all input pole vectors.
    ev, vl = spla.eig(A, left=True, right=False)
    u_p = []
    for i in range(vl.shape[1]):
        vli = np.asmatrix(vl[:,i]) 
        u_p.append(B.H*vli.T) 
    state_control = not any(np.linalg.norm(x) == 0.0 for x in u_p)

    # compute the controllability matrix
    c_plus = [A**n*B for n in range(A.shape[1])]
    control_matrix = np.hstack(c_plus)

    return state_control, u_p, control_matrix


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
    state_obsr = not any(np.sum(x)==0.0 for x in out_pole_vec)

    # compute observability matrix
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

    return state_control and state_obsr


def pole_zero_directions(G, vec):
    """    
    Parameters
    ----------
    G : numpy matrix
        The transfer function G(s) of the system.
    vec : numpy array
        A vector containing all the transmission poles or zeros of a system.
    
    Returns
    -------
    pz_dir : numpy array
        Pole or zero direction in the form - 
        (pole/zero, input direction, output direction)
        
    Note
    ----
    This method is going to give dubious answers if the function G has pole
    zero cancellation.        
    """
    
    pz_dir = []
    for d in vec:
        g = G(d)

        U, S, V = np.linalg.svd(g)
        V = np.transpose(np.conjugate(V))
        u_rows, u_cols = np.shape(U)
        v_rows, v_cols = np.shape(V)
        yz = np.hsplit(U, u_cols)[-1]
        uz = np.hsplit(V, v_cols)[-1]
        pz_dir.append((d, uz, yz))
        
    return pz_dir


def zero(A, B, C, D):
    """
    Computes the zeros of a transfer function in state space form.
    Parameters: A, B, C, D state space matrices
    Returns: zero vector (which you may use in my other functions)
    """
    z = sp.Symbol('z')
    top = np.hstack((A,B))
    bot = np.hstack((C,D))
    m = np.vstack((top, bot))
    M = sp.Matrix(m)
    [rowsA, colsA] = np.shape(A)
    [rowsB, colsB] = np.shape(B)
    [rowsC, colsC] = np.shape(C)
    [rowsD, colsD] = np.shape(D)
    p1 = np.eye(rowsA)
    p2 = np.zeros((rowsB, colsB))
    p3 = np.zeros((rowsC, colsC))
    p4 = np.zeros((rowsD, colsD))
    top = np.hstack((p1,p2))
    bot = np.hstack((p3,p4))
    p = np.vstack((top, bot))
    Ig = sp.Matrix(p)
    zIg = z*Ig
    f = zIg-M
    zf = f.det()
    zs = sp.solve(zf, z)
    
    return zs
