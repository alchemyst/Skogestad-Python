import sympy
import numpy
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')

s, t = sympy.symbols('s, t')
G = 3*(-2*s+1)/((10*s+1)*(5*s+1))
#Controller gains
Kc= [sympy.Rational(1/2), sympy.Rational(3/2), sympy.Rational(5/2), sympy.Rational(3)]
#Closed loop step response transfer function (closed loop * 1/s)
Y=[sympy.simplify(i*G/(s*(1+i*G))) for i in Kc]
#step response in time domain
y=[sympy.inverse_laplace_transform(i, s, t) for i in Y]

t = numpy.linspace(0.001, 50, num=200)
yt = [[numpy.real(complex(i.subs(t, j).n())) for j in t ] for i in y]

axes = plt.gca()
axes.set_ylim([-0.5, 2.5])
axes.set_xlabel('Time[sec]')

for i in ytm:
    plt.plot(tm, i)

leg = ["Kc = " + str(float(i)) for i in Kc]
plt.legend(leg, loc = 6,  bbox_to_anchor=(0.85, 0.9))