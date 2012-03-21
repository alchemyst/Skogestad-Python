'''
Created on Jan 27, 2012

@author: Carl Sandrock
'''

import numpy

#import control
#tf = control.TransferFunction

def circle(cx, cy, r):
    npoints = 100
    theta = numpy.linspace(0, 2*numpy.pi, npoints)
    y = cx + numpy.sin(theta)*r
    x = cx + numpy.cos(theta)*r
    return x, y

def distance_from_nominal(w, k, tau, theta, nom_response):
    r = k/(tau*w*i + 1)*numpy.exp(-theta*w*i)
    return numpy.abs(r - nom_response)

def arrayfun(f, A):
    """ recurses down to scalar elements in A, then applies f, returning lists containing the result"""
    if len(A.shape) == 0:
        return f(A)
    else:
        return [arrayfun(f, b) for b in A]

def listify(A):
    return [A]

def gaintf(K):
    r = tf(arrayfun(listify, K), arrayfun(listify, numpy.ones_like(K)))

def findst(G, K):
    """ Find S and T given a value for G and K """
    L = G*K
    I = numpy.eye(G.outputs, G.inputs)
    S = inv(I + L)
    T = S*L
    return S, T

def phase(G, deg=False):
    return numpy.unwrap(numpy.angle(G, deg=deg), discont=180 if deg else numpy.pi)

