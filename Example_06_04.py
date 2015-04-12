import numpy as np

from utils import poles, zeros, pole_zero_directions, BoundKS
from reporting import display_export_data


def Gs(s):
    return np.matrix([[(s - 2.5) / (s - 2), -(0.1 * s + 1) / (s + 2)],
                      [(s - 2.5) / (0.1 * s + 1), 1]])

# confirming the poles and zeros
# TODO this solution is clearly not correct and must be fixed
p = poles(Gs)
z = zeros(Gs)
print 'Poles: {0}'.format(p)
print 'Zeros: {0}'.format(z)

pdir = pole_zero_directions(Gs, [2], 'p')
display_export_data(pdir, 'Poles', ['u', 'y'], separator='|')

print '||KS|| > {:.3}'.format(BoundKS(Gs, [2]))
