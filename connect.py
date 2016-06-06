# -*- coding: utf-8 -*-
"""
Created on Wed May 22 19:03:56 2013

@author: Simon Streicher
"""
from __future__ import print_function
from utils import tf, feedback, tf_step
import matplotlib.pyplot as plt
import utilsplot

"""
This function aims to be the Python equivalent of the MatLab connect function
Reference:
http://www.mathworks.com/help/control/examples/connecting-models.html

The example used here is the same as in the reference
"""

# First example in reference to demonstrate working of tf object

# Define s as tf to enable its use as a variable
s = tf([1, 0])

# Define the transfer functions
F = tf(1, [1, 1])
G = tf(100, [1, 5, 100])
C = 20*(s**2 + s + 60) / s / (s**2 + 40*s + 400)
S = tf(10, [1, 10])

T = F * feedback(G*C, S)
# This is NOT the same figure as in the reference
t, y = tf_step(T, 6)
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.show()
#utilsplot.step(T, t_end = 6)

# Assign names to lines as in second example
F.u = 'r';  F.y = 'uF'
C.u = 'e';  C.y = 'uC'
G.u = 'u';  G.y = 'ym'
S.u = 'ym'; S.y = 'y'

# There must be a better way to call the name of an object if the object has
# already been assigned a name....
F.name = 'F'
C.name = 'C'
G.name = 'G'
S.name = 'S'

print(S)

def connect(blocks, line_in, line_out):
    """
    Returns the transfer function between line_in and line_out calculated
    by taking all the relationships in the blocks into account.
    blocks is a list of tf and sumblk objects
    line_in and line_out are line objects
    """
    pass


class sumblk(object):
    """
    Summation block diagram object allowing blocks to be added together
    based on input and output names
    """

    def __init__(self, name, out_name, in1, in2, add=True):
        """
        Initialize the sumblock from output and input lines
        Assumes addition by default
        """
        self.name = name
        self.out_name = out_name
        self.in1 = in1
        self.in2 = in2
        self.add = add

    def __repr__(self):
        if self.add:
            sign = ' + '
        else:
            sign = ' - '
        r = str("sumblk(" + str(self.out_name) + " = " + str(self.in1.name) +
                sign + str(self.in2.name) + ")")
        return r


class line(object):
    """
    Object representing a line in the block diagram
    To be used in the tf and sumblk classes
    """

    def __init__(self, name, upstream_conn, downstream_conn):
        """
        Initializes line object with name and connections where connections
        are tf objects
        """
        self.name = name

        self.up = upstream_conn
        self.down = downstream_conn

    def __repr__(self):
        r = str("line(" + str(self.up.name) + " --> "
                + str(self.down.name) + ")")
        return r

# Testing basic line and sumblock class structures

in1 = line('a', F, C)
in2 = line('b', C, G)
sum1 = sumblk('sum1', 'c', in1, in2)

print(in1)
print(in2)
print(sum1)
