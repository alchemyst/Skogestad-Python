import numpy as np
import scipy as sc
import scipy.linalg as sc_lin
import matplotlib.pyplot as plt

tau=0.5

def G(s):
        '''process transfer matrix'''
        G = np.multiply(1/(tau*s+1),np.matrix([[-87.8, 1.4], [-108.2, -1.4]]))
        return G

def K(s):
    '''controller'''
    K = np.multiply((tau*s+1)/s,np.matrix([[-0.0015, 0], [0, -0.075]]))
    return K

def w_I(s):
    '''magnitude multiplicative uncertainty in  each input'''
    w_I=(s+0.2)/(0.5*s+1)
    return w_I

def W_I(s):
    '''magnitude multiplicative uncertainty in each input'''
    W_I=np.matrix([[(s+0.2)/(0.5*s+1),0],[0,(s+0.2)/(0.5*s+1)]])
    return W_I

def M(s):
    M=np.multiply(w_I(s),K(s)*G(s)*sc_lin.inv(np.identity(np.size(K(s)*G(s),0))+K(s)*G(s)))
    return M                                          

def T_I(s):
    T_I=K(s)*G(s)*sc_lin.inv(np.identity(np.size(K(s)*G(s),0))+K(s)*G(s))
    return T_I

frequency=np.logspace(-3, 2,1000)
s=1j*frequency

max_singular_value_of_T_I=[]
mu_delta_T_I=[]

for si in s:
    [S1,V1,D1]=np.linalg.svd(T_I(si))   
    max_singular_value_of_T_I.append(np.max(V1))
    mu_delta_T_I.append(np.max(np.abs((np.linalg.eigvals(T_I(si))))))


print 'Robust Stability (RS) is attained if mu(T_I(jw)) < 1/|w_I(jw)| for all applicable frequency range'

plt.loglog(frequency,max_singular_value_of_T_I,'b')
plt.loglog(frequency,1/np.abs(w_I(s)),'r')
plt.loglog(frequency,mu_delta_T_I,'y')
plt.legend(('max_singular_value_of(T_I)','1/|w_I(jw)|','mu(T_I)'),'best', shadow=False)
plt.xlabel('frequency')
plt.ylabel('Magnitude') 
plt.show()