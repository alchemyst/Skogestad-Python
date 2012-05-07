import numpy as np 
import matplotlib.pyplot as plt 
import scipy.linalg as sc_lin 
from CBT_projek_sys_model import sys

#multiplicative input uncertainty
#inputs to this program is the weighting functions, system transfer function matrix and the controller function

#the outputs is the checks for NP 
#for unstructured uncertainty matrix , the RP and RS are tested.

def G(s):
    G=sys(s)[0]
    return G

def Wi(s):
    Wi=np.matrix([[16.8*((1/0.006)*s+1)**0.7/((1/0.00013)*s+1)**0.7, 0, 0], [0, 16.8*((1/0.006)*s+1)**0.7/((1/0.00013)*s+1)**0.7, 0], [0, 0, 16.8*((1/0.006)*s+1)**0.7/((1/0.00013)*s+1)**0.7]])
    return Wi


def Wp(s):
    Wp=(s/2+1.01E-1)/(s)
    Wp=np.matrix([[Wp, 0, 0], [0, Wp, 0], [0, 0, Wp]])
    return Wp


def K(s,Kc):
    ti=5000
    K=Kc*(1+1/(ti*s))
    K=np.matrix([[K, 0, 0], [0, 0, K], [0, K, 0]])
    return K

def Uncertainty(s):
    #the maximum singular value of this matric must always be less than 1 at all freqeuncies.
    [rows, columns]= np.shape(G(0.001))
    delta =0.2*np.ones([rows, columns])
    return delta


def P(s):
    [rows, columns]=np.shape(G(0.0001))

    P11 = np.zeros([rows*2, columns*2])
    P12 = np.zeros([rows*2, columns])
    P21 = np.zeros([rows, columns*2])
    P22 = np.zeros([rows, columns])

    P11[rows:2*rows, 0:columns]=Wp(s)*G(s)
    P11[rows:2*rows, columns:2*columns]=Wp(s)

    P12[0:rows, 0:columns]=Wi(s)
    P12[rows:2*rows, 0:columns]=Wp(s)*G(s)

    P21[0:rows, 0:columns]=-1*G(s)
    P21[0:rows, columns:2*columns]=np.identity(columns)

    P22=-1*G(s)

    return P11, P12, P21, P22



def N(s, Kc):
    columns = np.shape(G(0.001))[1]

    N11 = -1*Wi(s)*K(s, Kc)*G(s)*sc_lin.inv(np.identity(columns)+K(s, Kc)*G(s))
    N12 = -1*Wi(s)*K(s, Kc)*sc_lin.inv(np.identity(columns)+G(s)*K(s, Kc))
    N21 = Wp(s)*G(s)*sc_lin.inv(np.identity(columns)+K(s, Kc)*G(s))
    N22 = Wp(s)*sc_lin.inv(np.identity(columns)+G(s)*K(s, Kc))

    return N11, N12, N21, N22



def M(s, Kc): 
    M = N(s, Kc)[0]
    #this coresponds to the N11 of the N matrix
    return M 

#def NS():
#    """checking for nominal stability"""

print np.max(N(0.0000001, 0.00001)[3])

def NP(w_start, w_end, save=0):

    w=np.logspace(w_start, w_end, 100)
    Kc =[0.01, 0.001, 0.0001, 0.00001, 0.000001]

    N22=np.array([np.max(N(1j*w_i, Kc_i)[3]) for w_i in w for Kc_i in Kc])
    w_plot = np.array([w_i for w_i in w for Kc_i in Kc])

    plt.figure(1)
    plt.loglog(w_plot, N22, '+')
    plt.show()

    if save ==1:
        np.savetxt('NP N22 max norm', N22)
        np.savetxt('NP freq', w_plot)


def RS_RP_unstructured(w_start, w_end, save=0):
    w=np.logspace(w_start, w_end, 100)
    Kc =[0.01, 0.001, 0.0001, 0.00001, 0.000001]

    specteral_radius = np.array([np.max(np.abs(sc_lin.eigvals(M(1j*w_i, Kc_i)))) for w_i in w for Kc_i in Kc])
    max_sing_M = np.array([sc_lin.svd(M(1j*w_i, Kc_i))[1][0] for w_i in w for Kc_i in Kc])

    w_plot = np.array([w_i for w_i in w for Kc_i in Kc])
    plt.figure(2)
    plt.loglog([w[0], w[-1]], [1, 1], 'r')
    plt.loglog(w_plot, specteral_radius, 'r+')
    plt.loglog(w_plot, max_sing_M, 'b+')

    if save==1:
        np.savetxt('RP RS M unstructure specteral radius', specteral_radius)
        np.savetxt('RP RS M unstructure max sngular value', max_sing_M)
        np.savetxt('RP RS M freq', w_plot)

    plt.show()

NP(-4, 4, 1)
#RS_RP_unstructured(-4, 4, 1)
    
#def RP_unstructured():
    


