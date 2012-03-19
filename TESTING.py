import numpy as np
import scipy as sc 
import matplotlib.pyplot as plt 
import itertools as itr 

w   = np.logspace(-5,3,1000)
s   = 1j*w

def G(s):
    # the matrix transfer function
    G=[[0.001*np.exp(-5*s)*(-34.54*(s+0.0572))/((s+1.72E-04)*(4.32*s+1)),
    0.001*np.exp(-5*s)*(1.913)/((s+1.72E-04)*(4.32*s+1))],
    [0.001*np.exp(-5*s)*(-32.22*s)/((s+1.72E-04)*(4.32*s+1)),
    0.001*np.exp(-5*s)*(-9.188*(s+6.95E-04))/((s+1.72E-04)*(4.32*s+1))]]
    return G

freqresp = map (G,s)

sigmas=np.array([Sigma for U, Sigma,V in map(np.linalg.svd,freqresp)])

#maximum input and output vectors 
Umax=np.array([U[:,0] for U,sigma , V in map(np.linalg.svd,freqresp)])
Vmax=np.array([V[:,0] for U, sigma ,V in map(np.linalg.svd,freqresp)])

#minimum input and output vectors 
Umin=np.array([U[:,-1] for U,sigma , V in map(np.linalg.svd,freqresp)])
Vmin=np.array([V[:,-1] for U, sigma ,V in map(np.linalg.svd,freqresp)])

def Sing(w_s):
    s=1j*w_s
    return np.max(np.linalg.svd(G(s))[1])-1

sc.optimize.fsolve(Sing,0.1)

plt.loglog(w,sigmas)
plt.show()
