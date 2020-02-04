import numpy as np

from robustcontrol import pole_zero_directions, tf, mimotf, BoundKS, BoundST
from reporting import display_export_data

s = tf([1, 0], 1)

z = 2.5
p = 2

G = np.matrix([[(s - z)/(s - p), -(0.1*s + 1)/(s - p)],[(s - z)/(0.1*s + 1), 1]])
G = mimotf(G)

p_list = G.poles()
z_list = G.zeros()
print('Poles: {0}'.format(p_list))
print('Zeros: {0}\n'.format(z_list))

zerodata = pole_zero_directions(G, z_list, 'z')
poledata = pole_zero_directions(G, p_list, 'p')
rowhead = ['   u', '   y', '   e ']
display_export_data(zerodata, 'Zeros', rowhead)
display_export_data(poledata, 'Poles', rowhead)
up = poledata[0][1]

# Obtain stable plant
p = -2
Gs = np.matrix([[(s - z)/(s - p), -(0.1*s + 1)/(s - p)],[(s - z)/(0.1*s + 1), 1]])
Gs = mimotf(Gs)

#||KS||inf MSmin, MTmin using Utils functions
print('||KS||inf >= {:.2f}'.format(BoundKS(Gs, p_list, up)))
print('M_S,min = M_T,min = {:.2f}'.format(BoundST(G, p_list, z_list)))







