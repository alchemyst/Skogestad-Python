from __future__ import print_function
import numpy as np

from utils import poles, zeros, tf, mimotf

s = tf([1,0],[1])
G =  1 / (s + 2) * mimotf([[s - 1,  4],[4.5, 2 * (s - 1)]])
print('Poles: ' , poles(G))
print('Zeros: ' , zeros(G))
