import numpy as np
import scipy as sc
import scipy.signal as scs
import matplotlib.pyplot as plt 


def G(w):
    """ function to create the matrix of transfer functions"""
    """ these functions is for the process""" 
    s=w*1j
    
    """ the matrix transfer function""" 
    G=[[1/(s+1),1/(10*s+1)**2],[0.4/((s)*(s+3)),-0.1/(s**2+1)]]
    return G

def Controller(w):
    """ function to create the matrix of transfer functions"""
    """ these transfer functions is for the controllers""" 
    s=w*1j
    
    """ matrix transfer function"""
    K=[[1,0],[0,1]]
    return K
    
def SVD(w_start, w_end):
    w=np.logspace(w_start,w_end,1000)
    A=G(0.001)
    [x,y]=A.shape