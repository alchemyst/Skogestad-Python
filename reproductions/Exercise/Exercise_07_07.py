import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

def Gn(s):
    return 3*np.exp(-0.1*s)/((2*s + 1)*(0.1*s + 1)**2)

def Gp(s):
    return 3/(2*s + 1)

def lA(Gn, Gp):
    return np.abs(Gn - Gp)

def WA(s):
    return 0.9*s/((2*s + 1)*(0.1*s + 1))

def K(k,s):
    return k/s

omega = np.logspace(-2, 2, 500)

plt.figure()
plt.loglog(omega, lA(Gn(1j*omega), Gp(1j*omega)), 'b', label='lA(jw)')
plt.loglog(omega, WA(1j*omega), 'r', label='WA(jw)')
plt.title('Additive uncertainty weight')
plt.xlabel('Frequency [rad/s]', fontsize=14)
plt.ylabel('Magnitude', fontsize=14)
plt.legend(loc=2)
plt.show()

# RS condition |T|<|G|/|wA|
def RScondition(k):
    abs_ineq = [abs(Gp(1j*w)*K(k,1j*w)/(1+Gp(1j*w)*K(k,1j*w))) - abs(Gp(1j*w))/abs(WA(1j*w)) for w in omega]
    max_ineq = max(abs_ineq)
    return max_ineq

kcal = fsolve(RScondition,1)
print('The maximum value of k that yields stability is = ',np.round(kcal,2))


