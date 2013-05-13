'''
Created on 27 Mar 2013

@author: St Elmo Wilken
'''
import MIMO_Tools as mimo

# The system
A = [[-2, -2],
     [0, -4]]
B = [[1],
     [1]]
C = [[1, 0]]
D = [[0]]

control, in_vecs, c_matrix = mimo.state_controllability(A, B)


print "State Controllable: " + str(control)
print "Input Pole Vectors: " +str(in_vecs[0]) + " and " + str(in_vecs[1])
print "The controllability matrix is: "
print c_matrix



