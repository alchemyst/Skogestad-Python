import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import scipy.signal as scs

w = np.logspace(-5,2,1000)
def G(w):
    s = 1j*w

    G = [[((0.01*np.exp(-5*s))*(-34.54*(s+0.0572)))/((s + 1.72E-04)*(4.32*s + 1)), 
        ((0.01*np.exp(-5*s))*1.913)/((s + 1.72E-04)*(4.32*s + 1))],
        [((0.01*np.exp(-5*s))*(-30.22*s))/((s + 1.72E-04)*(4.32*s + 1)), 
         ((0.01*np.exp(-5*s))*(-9.188*(s + 6.95E-04)))/((s + 1.72E-04)*(4.32*s + 1))]]
    return G

A = G(0.01)
Ainv = np.linalg.pinv(A)
RGA1 = np.multiply(A,np.transpose(Ainv))
print RGA1
"""plotting the absolute value of each element
  RGA rule: lambda11 + lambda12 = 1
            lambda11 + lambda21 = 1
    therefore; lambda12 = lambda21  always"""
print abs(RGA1[0,0])
print abs(RGA1[0,1])
plt.plot(0.01,abs(RGA1[0,0]),'*',0.01,abs(RGA1[0,1]),'o')
plt.show()

"""Therefore, write a function that takes a value of frequency,
   calculate the inverse of the transfer function matrix
   calculate the RGA at that particular frequency
   calculate the absolute values of the 1st and 2nd elements of the RGA matrix
   plot the absolute values for that particular frequency"""  

store1 = []
store2 = []
for w_i in w:
    B = G(w_i)
    Binv = np.linalg.pinv(B)
    RGA_w = np.multiply(B,np.transpose(Binv))
    a = abs(RGA_w[0,0])
    b = abs(RGA_w[0,1])
    store1.append(a)
    store2.append(b)


print len(store1),len(store2)
plt.loglog(w,store1,'r-',w,store2,'b-')
plt.show()    
