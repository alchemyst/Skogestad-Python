import numpy as np
import scipy as sc
import scipy.signal as scs
import matplotlib.pyplot as plt
from utils import phase


# this function is different that the previous one in its inputs it takes
# this is easier than geving the actaul transfer function
"""due to the ease of which the roots of the transfer function could be determined from the numerator and denominators' polynomial
expansion"""


def G():
    """polynomial coefficients in the denominator and numerator"""
    Pz = [40]
    Pp = [1, 2, 1]
    return Pz, Pp


def Gm():
    """measuring elements dyanmics"""
    Pz = [1]
    Pp = [1, 1]
    return Pz, Pp


def Time_Delay():
    """matrix with theta values, combined time delay of system and measuring element"""
    Delay = [1]
    return Delay


def Gd():
    """polynomial coefficients in the denominator and numerator"""
    Pz = [8]
    Pp = [1, 1]
    return Pz, Pp


def RULES(R, wr):


    # rule 1 wc>wd

    def Gd_mod_1(w):
        return np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1])-1

    wd = sc.optimize.fsolve(Gd_mod_1, 10)

    wc_min_1 = wd


    # rule 2



    # rule 3



    # for perfect control

    def G_Gd_1(w):
        f = scs.freqs(G()[0], G()[1], w)[1]
        g = scs.freqs(Gd()[0], Gd()[1], w)[1]
        return np.abs(f)-np.abs(g)

    w_G_Gd = sc.optimize.fsolve(G_Gd_1, 0.001)

    plt.figure(1)
    if np.abs(scs.freqs(G()[0], G()[1], [w_G_Gd+0.0001])[1])>np.abs(scs.freqs(Gd()[0], Gd()[1], [w_G_Gd+0.0001])[1]):
        print "Acceptable control"
        print "control only at high frequencies", w_G_Gd, "< w < inf"

        w = np.logspace(-3, np.log10(w_G_Gd), 100)
        plt.loglog(w, np.abs(scs.freqs(G()[0], G()[1], w)[1]), 'r')
        plt.loglog(w, np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1]), 'r.')

        max_p = np.max([np.abs(scs.freqs(G()[0], G()[1], w)[1]), np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1])])

        w = np.logspace(np.log10(w_G_Gd), 5, 100)
        plt.loglog(w, np.abs(scs.freqs(G()[0], G()[1], w)[1]), 'b')
        plt.loglog(w, np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1]), 'b.')

        min_p = np.min([np.abs(scs.freqs(G()[0], G()[1], w)[1]), np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1])])

    if np.abs(scs.freqs(G()[0], G()[1], [w_G_Gd-0.0001])[1])>= np.abs(scs.freqs(Gd()[0], Gd()[1], [w_G_Gd-0.0001])[1]):
        print "Acceptable control"
        print "control up to frequency 0 < w < ", w_G_Gd

        w = np.logspace(-3, np.log10(w_G_Gd), 100)
        plt.loglog(w, np.abs(scs.freqs(G()[0], G()[1], w)[1]), 'b')
        plt.loglog(w, np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1]), 'b.')

        max_p = np.max([np.abs(scs.freqs(G()[0], G()[1], w)[1]), np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1])])

        w = np.logspace(np.log10(w_G_Gd), 5, 100)
        plt.loglog(w, np.abs(scs.freqs(G()[0], G()[1], w)[1]), 'r')
        plt.loglog(w, np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1]), 'r.')


        min_p = np.min([np.abs(scs.freqs(G()[0], G()[1], w)[1]), np.abs(scs.freqs(Gd()[0], Gd()[1], w)[1])])


    plt.loglog(w_G_Gd*np.ones(2), [max_p, min_p], 'g')


    # rule 4



    # rule 5
    # critical freqeuncy of controller needs to smaller than
    wc_5 = Time_Delay()[0]/2.0000


    # rule 6
    # control over RHP zeros

    Pz_G_Gm = np.polymul(G()[0], Gm()[0])

    if len(Pz_G_Gm) == 1:
        wc_6 = wc_5
    else:
        Pz_roots = np.roots(Pz_G_Gm)
        print Pz_roots
        if np.real(np.max(Pz_roots))>0:

            if np.imag(np.min(Pz_roots)) == 0:
                # it the roots aren't imagenary
                # looking for the minimum values of the zeros = > results in the tightest control
                wc_6 = (np.min(np.abs(Pz_roots)))/2.000

            else:
                wc_6 = 0.8600*np.abs(np.min(Pz_roots))

        else:
            wc_6 = wc_5





    # rule 7


    def G_GM(w):
        Pz = np.polymul(G()[0], Gm()[0])
        Pp = np.polymul(G()[1], Gm()[1])
        G_w = scs.freqs(Pz, Pp, w)[1]
        return np.abs(phase(G_w))-np.pi

    w = np.logspace(-3, 3, 100)
    plt.figure(2)
    Pz = np.polymul(G()[0], Gm()[0])
    Pp = np.polymul(G()[1], Gm()[1])
    [w, h] = scs.freqs(Pz, Pp, w)

    plt.subplot(211)
    plt.loglog(w, np.abs(h))
    plt.subplot(212)
    plt.semilogx(w, phase(h))


    wc_7 = np.abs(sc.optimize.fsolve(G_GM, 10))

    w_vec = [wc_5, wc_6, wc_7]

    wc_min_everything = np.min(w_vec)

    print "  "
    print "maximum value of wc < ", wc_min_everything





    # rule 8
    # unstable RHP poles
    Poles_p = np.roots(G()[1])

    vec_p = [wc_min_1]

    for p in Poles_p:
        if np.real(p) > 0:
            vec_p.append(2*np.abs(p))


    wc_min_everything = np.max(vec_p)

    print "minimum value of wc > ", wc_min_everything

    plt.show()

RULES(1, 8)
