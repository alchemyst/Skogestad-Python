
"This code plots figure 5.15 of Example 5.12..."

import numpy as np
import matplotlib.pyplot as plt


w = np.logspace(-3, 3, 1000)
s = w*1j


def G(s):    
    G = 40/((5*s+1)*(2.5*s+1))
    return G

def Gd(s):    
    Gd = 3*((50*s+1)/((s+1)*(10*s + 1)))
    return Gd


plt.figure(1)
plt.loglog(w, [abs(G(thisw)) for thisw in w])
plt.loglog(w, [abs(Gd(thisw)) for thisw in w])
plt.loglog(w, np.ones(len(w)), '-.')
plt.title('|G| & |Gd| over frequency')
plt.xlabel('frequency [rad/s]')
plt.ylabel('Magnitude') 
plt.show()    


