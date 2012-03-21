import numpy as np
import scipy as sc
import scipy.signal as scs
import matplotlib.pyplot as plt
from utils import phase

# Loop shaping is an iterative procedure where the designer
# 1. shapes and reshapes |L(jw)| after computing PM and GM,
# 2. the peaks of closed loop frequency responses (Mt and Ms),
# 3. selected closed-loop time responses,
# 4. the magnitude of the input signal
#
# 1 to 4 are the important frequency domain measures used to assess
#   perfomance and characterise speed of response

w = np.logspace(-2, 1, 1000)
s = w*1j
kc = 0.05
#plant model
G = 3*(-2*s+1)/((10*s+1)*(5*s+1))
#Controller model
K = kc*(10*s+1)*(5*s+1)/(s*(2*s+1)*(0.33*s+1))
#closed-loop transfer function
L = G*K

#magnitude and phase
plt.subplot(2, 1, 1)
plt.loglog(w, abs(L))
plt.loglog(w, 1*np.ones(len(w)))
plt.ylabel('Magnitude')

plt.subplot(2, 1, 2)
plt.semilogx(w, phase(L, deg=True))
plt.semilogx(w, (-180)*np.ones(len(w)))
plt.ylabel('Phase')
plt.xlabel('frequency (rad/s)')
plt.show()

# From the figure we can calculate GM and PM,
# cross-over frequency wc and w180
# results:GM = 1/0.354 = 2.82
#         PM = -125.3 + 180 = 54 degC
#         w180 = 0.44
#         wc = 0.15
