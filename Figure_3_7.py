import numpy as np
import numpy.linalg as la
from matplotlib import pyplot


def G1(s):
    """def the function for Figure 3.7)a the distilation process"""
    return 1/(75*s+1)*np.array([[87.8, -86.4], [108.2, -109.6]])

def G2(s):
    """def the function for Figure 3.7)b the spinning satelite"""
    return 1/(s**2+10**2)*np.array([[s-10**2, 10*(s+1)], [-10*(s+1), s-10**2]])

def SVD(G, s):
    """Function to determine singular values"""
    freqresp = map(G, s)
    sigmas = np.matrix([Sigma for U, Sigma, V in map(la.svd, freqresp)])
    return sigmas
    
def condition_number(G, s):
    """Function to determine condition number"""
    freqresp = map(G, s)
    sigmas = np.matrix([Sigma for U, Sigma, 
                           V in map(np.linalg.svd, freqresp)])
    nrows, ncols = sigmas.shape
    gamma = np.zeros(nrows)
    for i, row_vector in enumerate(sigmas): 
        gamma[i] = sigmas[i,0]/sigmas[i,1] 
    
    return gamma


"""Plotting of Figure 3.7)a and 3.7)b"""
pyplot.subplot(1, 2, 1)
w = np.logspace(-4, 1, 1000)
pyplot.loglog(w, SVD(G1, 1j*w))
pyplot.xlabel(r'Frequency [rad/s]', fontsize=14)
pyplot.ylabel(r'Magnitude', fontsize=15)
pyplot.title('Distillation process 3.7(a)')
pyplot.text(0.001, 220, r'$\bar \sigma$(G)', fontsize=15)
pyplot.text(0.001, 2, r'$\frac{\sigma}{}$(G)', fontsize=20)
pyplot.subplot(1, 2, 2)
pyplot.w = np.logspace(-2, 2, 1000)
pyplot.loglog(w, SVD(G2, 1j*w))
pyplot.xlabel(r'Frequency [rad/s]', fontsize=14)
pyplot.title('Spinning satellite 3.7(b)')
pyplot.text(2, 20, r'$\bar \sigma$(G)', fontsize=15)
pyplot.text(1, 0.2, r'$\frac{\sigma}{}$(G)', fontsize=20)
pyplot.show()

"""Plotting of Condition Numbers"""
pyplot.subplot(1, 2, 1)
w = np.logspace(-4, 1, 1000)
pyplot.semilogx(w, condition_number(G1, 1j*w))
pyplot.xlabel(r'Frequency [rad/s]', fontsize=14)
pyplot.ylabel(r'Magnitude', fontsize=15)
pyplot.title('Condition number for 3.7(a)')
pyplot.subplot(1, 2, 2)
pyplot.w = np.logspace(-2, 2, 1000)
pyplot.semilogx(w, condition_number(G2, 1j*w))
pyplot.xlabel(r'Frequency [rad/s]', fontsize=14)
pyplot.title('Condition number for 3.7(b)')
pyplot.show()