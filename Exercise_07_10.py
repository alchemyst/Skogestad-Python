"""Exercise 7.10 on page287 - Skogestad"""

import numpy as np
import matplotlib.pyplot as plt

def G(s):  # G detailed
    return 3*(-0.5*s + 1)/((2*s + 1)*(0.1*s + 1)**2)

def Gn(s):  # Nominal model
    return 3/(2*s + 1)

def li(G, Gn):  # Relative uncertainty
    return np.abs((G - Gn)/Gn)

def Wi1(s):  # Rational transfer function Wi1(s)
    return (-0.01*s**2 + 0.3)/(0.1*s + 1)**2

def Wi2(s):  # Rational transfer function Wi1(s) of higher order for better fit of li(s)
    return ((-0.01*s**2 + 0.3)/(0.1*s + 1)**2)*((0.124*s**2 + 9.008*s + 0.0016)/(0.112*s**2 + 1.339*s + 3.5))

w = np.logspace(-3, 3, 1000)
s = 1j*w

plt.figure(1)
plt.loglog(w, li(G(s), Gn(s)), 'b', label = 'li(jw)')
plt.loglog(w, np.abs(Wi1(s)), 'g--', label ='Wi1(jw)')  # Wi1(s) of low order (2)
plt.loglog(w, np.abs(Wi2(s)), 'r--', label = 'Wi2(jw)>=li(jw)')  # Wi2(s) of higher oder (4) for better fit of li(s)
plt.title('Wi(s) approximation of li(s)')
plt.xlabel('Frequency [rad/s]', fontsize=14)
plt.ylabel('Magnitude', fontsize=14)
plt.legend(loc=2)
plt.show()
