import sympy
from scipy import signal
import numpy
import matplotlib.pyplot as plt

s = sympy.Symbol('s')
G = 3*(-2*s+1)/((10*s+1)*(5*s+1))
K= 1.14*(1+1/(12.7*s))
#closed loop transfer function
Y= sympy.simplify(K*G/(1+K*G))
#Controller output transfer function
U = sympy.simplify(K/(1+K*G))
#Split into numerator and denominator
numden = [[sympy.poly(sympy.numer(Y)), sympy.poly(sympy.denom(Y))], [sympy.poly(sympy.numer(U)), sympy.poly(sympy.denom(Y))]]
#Get coefficients
numdenc = [[i.all_coeffs() for i in j] for j in numden]
#LTI models
[y, u] = [signal.lti([float(i) for i in numc], [float(j) for j in denc]) for numc, denc in numdenc]
#Step responses
t = numpy.linspace(0, 80, num=200)
yt = signal.lti.step(y, T=t)
ut = signal.lti.step(u, T=t)

axes = plt.gca()
axes.set_ylim([-0.5, 2.0])
axes.set_xlabel('Time[sec]')

plt.plot(yt[0], yt[1])
plt.plot(ut[0], ut[1])

leg = ["y(t)", "u(t)"]
plt.legend(leg, loc = 6,  bbox_to_anchor=(0.85, 0.9))
plt.show()