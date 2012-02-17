import numpy as np
import scipy as sc
import scipy.signal as scs
import matplotlib.pyplot as plt 


def Process():
    Gz=[[[1,2,3],1],[[1,2],[2,3,4,5]]]
    Gp=[[[2,3,4,5],[3,4]],[1,3,4,5,6],[34,3,4,2,6,54]]
    return Gz, Gp

def Controller():
    Gz=[[1,0],[0,1]]
    Gp=[[1,1],[1,1]]
    return Gz, Gp


def Closed_loop (Kz,Kp,Gz,Gp):
    """this function returns the polynomial constants of the closed loop transfer function's numerator and denominator"""
    """ this function is to take multivariable functions""" 
    
    """Kz is the polynomial constants in the numerator""" 
    """Kp is the polynomial constants in the denominator """
    """Gz is the polynomial constants in the numerator """
    """Gp is the polynomial constants in the denominator"""
    
    """ controller matrix needs to be the same size as systems"""
    """ needs to take square matrices""" 
    
    for x in range(len(Kz)):
        for y in range(len(Kz)):
    """calculating the product of the two polynomials in the numerator and denominator of transfer function GK"""
        Z_GK         =np.polymul(Kz[x][y]],Gz[x][y])
        P_GK         =np.polymul(Kp[x][y],Gp[x][y])    
    """calculating the polynomial of closed loop function T=(GK/1+GK)"""
        Zeros_poly   =Z_GK
        Poles_poly   =np.polyadd(Z_GK,P_GK)   
    return      Zeros_poly,Poles_poly

def combine(x):
    