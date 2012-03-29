import numpy as np
import scipy.linalg as sc_lin
import matplotlib.pyplot as plt


# start deriving and implementing equation for uncertainty


def uncertian_MIMO():

    #upper bounds on S'
    #eq6-88 pg 248

    #check input and output condition number along with RGA elements
