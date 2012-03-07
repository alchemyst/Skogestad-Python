import numpy as np 
import scipy as sc
import scipy.signal as scs
import matplotlib.pyplot as plt 


def G(s):
    """ give the transfer matrix of the system"""
    G=[[  ]]
    Gd=[[  ]]
    return G ,Gd

def Gms(s):
    """ stable, minimum phase system of G and Gd"""
    G_ms=[[]]
    Gd_ms=[[]]
    return G_ms, Gd_ms

def Zeros_Poles_RHP():
    """ Give a vector with all the RHP zeros and poles
    RHP zeros and poles are calulated from sage program"""
    
    Zeros_G     =[]
    Poles_G     =[]
    Zeros_Gd    =[]
    Poles_Gd    =[]
    return Zeros_G , Poles_G , Zeros_Gd , Poles_Gd 

def deadtime():
    """ vector of the deadtime of the system""" 
    dead_G      =[]
    dead_Gd     =[]
    return dead_G, dead_Gd 


def PEAK_MIMO():
    