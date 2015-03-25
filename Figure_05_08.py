from utils import tf
import utilsplot

s = tf([1, 0])

#Figure 5.8
G = (-s + 1)/(s + 1)
Kc = [0.1, 0.5, 0.9]
K = -1*(s)/((1 + 0.02*s)*(1 + 0.05*s))

utilsplot.freq_step_response_plot(G, K, Kc, 0.2, 1000, 'S', 'Figure 5.8')