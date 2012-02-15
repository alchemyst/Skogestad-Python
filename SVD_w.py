import numpy as np 
import scipy as sc
import scipy.signal as scs
import matplotlib.pyplot as plt 


def G(w):
    """ function to create the matrix of transfer functions"""
    s=w*1j
    
    """ the matrix transfer function""" 
    G=[[1/(s+1),1/(10*s+1)**2],[0.4/((s)*(s+3)),-0.1/(s**2+1)]]
    return G
  
  
  
def SVD_w(w_start, w_end):
    w=np.logspace(w_start,w_end,100000)
    store_max=np.zeros(len(w))
    store_min=np.zeros(len(w))
    count=0
    for w_iter in w:
        A=G(w_iter)
        [U,S,T]=np.linalg.svd(A)
        
        store_max[count]=S[0]
        store_min[count]=S[1]
        count=count+1
        
    """ plot of the singular values , maximum ans minimum"""   
    plt.subplot(211)  
    plt.loglog(w,store_max,'r')
    plt.loglog(w,store_min,'b')
    
    """ plot of the condition number""" 
    
    plt.subplot(212)
    plt.loglog(w,store_max/store_min)
    plt.show()

SVD_w(-3,3)