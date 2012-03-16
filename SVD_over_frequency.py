import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

w = np.logspace(-5, 1, 1000)
s = 1j*w

def G(s):
    return 0.01*np.exp(-5*s)/((s + 1.72e-4)*(4.32*s + 1))*np.array([[-34.54*(s + 0.0572), 1.913],
                                                                    [-30.22*s, -9.188*(s + 6.95e-4)]])
    
freqresp = map(G, s)
# Note: The above map statement is equivalent to the following two forms
# Using a list comprehension:
# freqresp = [G(thiss) for thiss in s]
# Using an explicit for loop
# freqresp = []
# for thiss in s:
#     freqresp.append(G(thiss))

sigmas = np.array([Sigma for U, Sigma, V in map(la.svd, freqresp)])

plt.loglog(w, sigmas)
plt.show()