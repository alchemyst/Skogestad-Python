import sympy
import numpy
from scipy import signal
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')

s = sympy.Symbol('s')
G = 3*(-2*s+1)/((10*s+1)*(5*s+1))
#Controller gains
Kc= [0.5, 1.5, 2.5, 3.0]
#Closed-loop transfer function
Y=[sympy.simplify(i*G/(1+i*G)) for i in Kc]
#Split into numerator and denominator, get coefficients
numden = [[sympy.poly(sympy.numer(i)), sympy.poly(sympy.denom(i))] for i in Y]
numdenc = [[i[0].all_coeffs(), i[1].all_coeffs()] for i in numden]
#lti model
ylti = [signal.lti([float(j) for j in i[0]], [float(k) for k in i[1]]) for i in numdenc]
t = numpy.linspace(0, 50, num=200)
#step response
yt = [signal.lti.step(i, T=t) for i in ylti]
axes = plt.gca()
axes.set_ylim([-0.5, 2.5])
axes.set_xlabel('Time[sec]')
for i in yt:
    plt.plot(i[0], i[1])
leg = ["Kc = " + str(i) for i in Kc]
plt.legend(leg, loc = 6,  bbox_to_anchor=(0.85, 0.9))

