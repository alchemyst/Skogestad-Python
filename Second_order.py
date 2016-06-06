from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scs


def Analyses_second_order(v, t, max_peeks):
    """take vector of output (v) , time vector along with the outputs (t) and  time where the analyses stops (t_end)
    scaled models are used in this analyses"""

    # empty vectors for storing data
    max_v = []
    min_v = []
    time_min = []
    time_max = []
    plt.plot(t, v)

    if np.max(v) >= 1:
        for k in range(max_peeks):
            if len(v)-2 >= 1:
                # give indices from min to max value of the output vector
                sort = np.argsort(v)
                # store the maximum value and time where the maximum occurs of the system
                max_v.append([v[sort[-1]]])
                # updating the vector to not contain information already contained in max vector
                v = v[sort[-1]+1:]
                # sorting updated output vector that doesn't contain the previous maximum value
                sort = np.argsort(v)
                # store the minimum value and time where the minimum occurs of the system
                min_v.append([v[sort[0]]])
                # updating the vector to not contain information already contained in the min value
                v = v[sort[0]+1:]

    else:
        return "not second order"

    # finding the times where the corresponding maximums and minimums occur
    for i in range(len(t)):
        for j in range(len(max_v)):

            if y[i] == max_v[j]:
                time_max.append(t[i])
            if y[i] == min_v[j]:
                time_min.append(t[i])

    max_v, min_v = np.matrix(max_v), np.matrix(min_v)

    # the analysis of rise time, overshoot decay ratio
    if max_peeks > 1:
        decay_ratio = (max_v[0, 0]-1)/(max_v[1, 0]-1)
        overshoot = max_v[0, 0]

    print('decay ratio = ', decay_ratio)
    print('overshoot = ', overshoot)

    total_varaince = np.sum(np.abs(np.diff(y)))
    print('Total Variance ', total_varaince)

    plt.plot(time_max, max_v, 'rD',label='maximum')
    plt.plot(time_min, min_v, 'bD',label='minimum')
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.legend()
    plt.show()

# example
# time constant = 1 dampening coefficient = 0.7 and steady state gain = 1
# G = 1/(s**2+1.4*s+1)

f = scs.lti([1], [1, 0.8, 1])
[t, y] = f.step()


#y = np.random.random(20)
#t = np.linspace(0, 19, 20)


Analyses_second_order(y, t, 2)
