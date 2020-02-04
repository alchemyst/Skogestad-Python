from robustcontrol import pole_zero_directions, tf, mimotf
from reporting import display_export_data

s = tf([1, 0])

G = 1/(s + 2)*mimotf([[s - 1, 4],
                      [4.5, 2*(s - 1)]])

# Poles and zeros calculated in Example 4.11

zerodata = pole_zero_directions(G, [4.], 'z')
poledata = pole_zero_directions(G, [-2.], 'p')
rowhead = ['   u', '   y', '   e ']

display_export_data(zerodata, 'Zeros', rowhead)
display_export_data(poledata, 'Poles', rowhead)
