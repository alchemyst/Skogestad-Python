'''
Created on 27 Mar 2013

@author: St Elmo Wilken
'''


import control as cn
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg as spla

"""
This toolbox assumes you have the control toolbox at your
disposal...
"""

def state_controllability(A, B):
    """
    Parameters: A => state space representation of matrix A
                B => state space representation of matrix B
    
    Returns: state_control (Bool) => True if state controllable
             in_pole_vec          => Input pole vectors for the states
             control_matrix       => State Controllability Matrix
    
    This method checks of the state space description of the system
    is state controllable according to Skogestad section 4.2.
    
    Note: The Gramian matrix type of solution has already been implemented by
    the Control Toolbox folks.
    """
    state_control = True
    
    # by computing all input vectors.
    ev, vl = spla.eig(A, left = True, right = False)
    rows, cols = np.shape(vl)
    in_pole_vec = [np.around(np.dot(np.transpose(np.conjugate(x)), B)[0][0],3)
           for x in np.hsplit(vl, cols)]
    num_zero = np.size(np.where(np.array(in_pole_vec) == 0.0)[0])
    if num_zero > 0: state_control = False
    
    # by computing the controllability matrix
    rows, cols = np.shape(A)
    c_plus = []
    for index in range(cols):
        c_plus.append(np.dot(np.linalg.matrix_power(A, index),B))
    control_matrix = np.hstack(c_plus)
    
    return state_control, in_pole_vec, control_matrix

    
