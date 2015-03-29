from utils import tf
from utilsplot import bode

G = tf([5], [10, 1], deadtime=2)

bode(G, -3, 1, 'Figure 2.2', True)
