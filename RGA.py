import numpy as np 
import matplotlib.pyplot as plt 


def RGA_SS(A):
    """ this is for steady state RGA""" 
    """ here is no freqeuncy dependency""" 
    
    """ A is a steady state gain matrix of system""" 
    
    RGA_SS=np.multiply(A,np.transpose(np.linalg.pinv(A)))
    print RGA_SS
    return RGA_SS 
    
"""example 3.9 Skogestad pg 85"""

A=np.matrix([[1,1],[0.4,-0.1]])

RGA=RGA_SS(A)



""" the next two function is to calculate the frequancy dependend RGA""" 

def G(w):
    """ function to create the matrix of transfer functions"""
    s=w*1j
    
    """ the matrix transfer function"""
    """ this spesific one is Example 3.11"""  
    G=[[0.001*np.exp(-5*s)*(-34.54*(s+0.0572))/((s+1.72E-04)*(4.32*s+1)),
        0.001*np.exp(-5*s)*(1.913)/((s+1.72E-04)*(4.32*s+1))],
        [0.001*np.exp(-5*s)*(-32.22*s)/((s+1.72E-04)*(4.32*s+1)),
        0.001*np.exp(-5*s)*(-9.188*(s+6.95E-04))/((s+1.72E-04)*(4.32*s+1))]]
    return G


def RGA_w(w_start,w_end,x,y):
    """ w_start is the start of logspace""" 
    """ w_end is the ending of the logspace""" 
    """ x and y is refer to the indices of the RGA matrix that needs to be plotted""" 
    
    """ this is to calculate the RGA at different freqeuncies """ 
    """ this give more conclusive values of which pairing would give fast responses""" 
    """ under dynamic situations""" 
    
    w=np.logspace(w_start,w_end,1000)
    store=np.zeros([len(x),len(w)])

    count=0
    for w_i in w:
        A=G(w_i)
        RGA_w=np.multiply(A,np.transpose(np.linalg.pinv(A)))
        store[:,count]=RGA_w[x,y]
        count=count+1

    for i in range(len(x)):
        plt.loglog(w,store[i,:])
    
    plt.title('RGA over Freq')
    plt.xlabel('w')
    plt.ylabel('|RGA values| gevin x ,y ')
    plt.show()

       
       
RGA_w(-6,2,[0,1],[0,0])
