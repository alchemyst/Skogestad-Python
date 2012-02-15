import numpy as np 



def RGA_SS(A):

    RGA_SS=np.multiply(A,np.transpose(np.linalg.pinv(A)))
      
    return RGA_SS 
    
"""example 3.9 Skogestad pg 85"""

A=np.matrix([[1,1],[0.4,-0.1]])

RGA=RGA_SS(A)
print RGA 


