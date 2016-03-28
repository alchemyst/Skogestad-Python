from __future__ import print_function
import numpy as np

from utils import poles, zeros

def G(s):
    return 1 / (s + 2) * np.matrix([[s - 1,  4],
                                    [4.5, 2 * (s - 1)]])
print('Poles: ' , poles(G))
print('Zeros: ' , zeros(G))
