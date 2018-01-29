import numpy as np
import matplotlib.pyplot as plt

w = np.logspace(-3, 3, 3000)
s = 1j*w
G = 3*(-2*s + 1)/((10*s + 1)*(5*s + 1))
Kc = 1.14
K = Kc*(12.7*s + 1)/(12.7 *s)
L = G * K
# L is a 1 x 1 matrix, and therefore the determinant is equal to the scalar
Ldet = 1 + L # for SISO

plt.figure('Figure 4.8')
plt.title('Typical Nyquist plot of det(I + L(jw))')
plt.plot(Ldet.real, Ldet.imag)
plt.plot(Ldet.conj().real, Ldet.conj().imag, '--')
plt.ylim(-3, 3.0)
plt.xlim(-2, 2.0)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
plt.xlabel('Re')
plt.ylabel('Im')
