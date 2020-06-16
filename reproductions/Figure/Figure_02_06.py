#The curve displayed by this code is only for the ultimate gain in Figure 2.6
import numpy
import matplotlib.pyplot as plt

G = (3/(-2*s +1))/((10*s+1)*(5*s+1))

#Parameters
r = 1
ω = 0.42
s = ω*1j
t = numpy.linspace(0, 50, 100)

#System gain
Kc = 2.5

#System phase
PS = numpy.angle(G)        #All the curves have the same phase

#System outputs
y = Kc*numpy.sin(ω*t + PS)

plt.figure('Figure 2.6')
plt.title('Effect of proportional gain Kc on the closed-loop response y(t) for the inverse response process')
plt.plot(t, y)
plt.xlabel('Time [s]')
plt.ylabel('System response')
plt.legend(['Kc=Ku=2.5'])
plt.show()
