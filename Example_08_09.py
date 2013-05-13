import numpy as np
import utils
import matplotlib.pyplot as plt
import scipy.optimize

I = np.identity(2)

def KG(s):
    Ggain = np.matrix([[-87.8, 1.4],
                       [-108.2, -1.4]])
    Kgain = np.matrix([[-0.0015, 0],
                       [0, -0.075]])
    return 1/s*Kgain*Ggain


def w_I(s):
    """magnitude multiplicative uncertainty in  each input"""
    return (s + 0.2) / (0.5 * s + 1)


def W_I(s):
    """magnitude multiplicative uncertainty in each input"""
    return np.matrix([[(s + 0.2) / (0.5 * s + 1), 0], 
                      [0, (s + 0.2) / (0.5 * s + 1)]])


def T_I(s):
    return KG(s) * (I + KG(s)).I


def M(s):
    return w_I(s) * T_I(s)


def maxsigma(G):
    return max(utils.sigmas(G))

def specrad(G):
    return max(np.abs(np.linalg.eigvals(G)))

def mu_ubound(G):
    """ Calcuate the scaled singular value by direct optimisation """
    def scaled_system(d):
        D = np.asmatrix(np.diag(d))
        return maxsigma(D*G*D.I)
    [dopt, minvalue, _, _, _] = scipy.optimize.fmin_slsqp(scaled_system, [1, 1], disp=False, full_output=True)
    return minvalue

omega = np.logspace(-3, 2, 1000)
s = 1j * omega

T_Is = map(T_I, s)

print 'Robust Stability (RS) is attained if mu(T_I(jw)) < 1/|w_I(jw)| for all applicable omega range'

plt.rc('text', usetex=True)
plt.loglog(omega, map(maxsigma, T_Is))
plt.loglog(omega, 1 / np.abs(w_I(s)))
plt.loglog(omega, map(specrad, T_Is))
plt.loglog(omega, map(mu_ubound, T_Is))
plt.legend((r'$\bar\sigma(T_I)$', 
            r'$\frac{1}{|w_I(jw)|}$', 
            r'$\rho(T_I)$',
            r'$\min_{D}\bar\sigma(DT_ID^{-1})$'), 'best')
plt.xlabel(r'$\omega$')
plt.ylabel('Magnitude')
plt.show()
