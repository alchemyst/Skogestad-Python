import numpy as np

from utils import pole_zero_directions
from reporting import display_export_data

def G(s):
    return 1 / (s + 2) * np.matrix([[s - 1,  4],
                                    [4.5, 2 * (s - 1)]])

# Poles and zeros calculated in Example 4.11

zerodata = pole_zero_directions(G, [4.], 'z')
poledata = pole_zero_directions(G, [-2.], 'p')
rowhead = ['u','y']

display_export_data(zerodata, 'Zeros', rowhead, separator='|')
display_export_data(poledata, 'Poles', rowhead, separator='|')
