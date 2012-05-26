import numpy as np
import scipy as sc
import scipy.linalg as sc_lin
import matplotlib.pyplot as plt

a=10
def G(s):
        '''process transfer matrix'''
        G = np.multiply(1/((s**2)+a**2),np.matrix([[s-a**2, a*(s+1)], [-a*(s+1), s-a**2]]))
        return G

def K(s):
    '''controller'''
    K = np.eye(np.size(G(s),0))
    return K                                      

def T(s): 
    '''this is a special case where T_I(s)= T(s) '''
    T=G(s)*K(s)*sc_lin.inv(np.identity(np.size(G(s)*K(s),0)) + G(s)*K(s))
    return T

frequency=np.logspace(-3, 2,1000)
s=1j*frequency

max_singular_value_of_T=[]
mu_T=[]

for si in s:
    [S1,V1,D1]=np.linalg.svd(T(si))   
    max_singular_value_of_T.append(np.max(V1))
    mu_T.append(np.max(np.abs((np.linalg.eigvals(T(si))))))

plt.loglog(frequency,max_singular_value_of_T,'b')
plt.loglog(frequency,mu_T,'r')
plt.legend(('max_singular_value(T)','mu(T)'), 'best', shadow=False)
plt.xlabel('frequency')
plt.ylabel('Magnitude')
plt.show()