import numpy as np
import scipy as sc
import scipy.linalg as sc_lin
import scipy.optimize as sc_opt
import scipy.signal as scs
import matplotlib.pyplot as plt


def G(s):
    """ give the transfer matrix of the system"""
    G = np.matrix([[1/s+1, 1], [1/(s+2)**2, (s+1)/(s-2)]])
    return G

def Gd(s):
    Gd = np.matrix([[1/(s+1)], [1/(s+20)]])
    return Gd

def reference_change():
    """Reference change matrix/vector for use in eq 6-52 pg 242 to check input saturation"""

    R = np.matrix([[1, 0], [0, 1]])
    R = R/np.linalg.norm(R, 2)
    return R

def Gms(s):
    """ stable, minimum phase system of G and Gd"""
    G_ms = [[]]
    Gd_ms = [[]]

    G_s = [[]]
    Gd_s = [[]]
    return G_ms, Gd_ms, G_s, Gd_s

def Zeros_Poles_RHP():
    """ Give a vector with all the RHP zeros and poles
    RHP zeros and poles are calculated from sage program"""

    Zeros_G = [1]
    Poles_G = [-2]
    Zeros_Gd = []
    Poles_Gd = []
    return Zeros_G, Poles_G, Zeros_Gd, Poles_Gd



def deadtime(s):
    """ vector of the deadtime of the system"""
    #individual time delays
    dead_G = []
    dead_Gd = []

    return dead_G, dead_Gd




def PEAK_MIMO(w_start, w_end, error_poles_direction, wr):
    """ this function is for multivariable system analysis of controllability
    gives:
    minimum peak values on S and T with or without deadtime
    R is the expected worst case reference change, with condition that ||R||2<= 2
    wr is the frequency up to where reference tracking is required"""


    def plot_direction(direction, name, color, figure_num):
        plt.figure(figure_num)
        if (direction.shape[0])>2:
            for i in range(direction.shape[0]):
                #label = '%s Input Dir %i' % (name, i+1)

                plt.subplot((direction.shape[0]), 1, i + 1)
                plt.title(name)
                plt.semilogx(w, direction[i, :], color)

        else:

            plt.subplot(211)
            plt.title(name)
            plt.semilogx(w, direction[0, :], color)

            plt.subplot(212)
            plt.title(name)
            plt.semilogx(w, direction[1, :], color)


    #importing most of the zeros and poles data
    [Zeros_G, Poles_G, Zeros_Gd, Poles_Gd] = Zeros_Poles_RHP()



    #just to save unnecessary calculations that is not needed
    #sensitivity peak of closed loop. eq 6-8 pg 224 skogestad

    if np.sum(Zeros_G)!= 0:
        if np.sum(Poles_G)!= 0:

            #two matrices to save all the RHP zeros and poles directions
            yz_direction = np.matrix(np.zeros([G(0.001).shape[0], len(Zeros_G)]))
            yp_direction = np.matrix(np.zeros([G(0.001).shape[0], len(Poles_G)]))


            for i in range(len(Zeros_G)):

                [U, S, V] = np.linalg.svd(G(Zeros_G[i]+error_poles_direction))
                yz_direction[:, i] = U[:, -1]


            for i in range(len(Poles_G)):
                #error_poles_direction is to to prevent the numerical method from breaking
                [U, S, V] = np.linalg.svd(G(Poles_G[i]+error_poles_direction))
                yp_direction[:, i] = U[:, 0]


            yz_mat1 = np.matrix(np.diag(Zeros_G))*np.matrix(np.ones([len(Zeros_G), len(Zeros_G)]))
            yz_mat2 = np.transpose(yz_mat1)

            Qz = (np.transpose(np.conjugate(yz_direction))*yz_direction)/(yz_mat1+yz_mat2)

            yp_mat1 = np.matrix(np.diag(Poles_G))*np.matrix(np.ones([len(Poles_G), len(Poles_G)]))
            yp_mat2 = np.transpose(yp_mat1)

            Qp = (np.transpose(np.conjugate(yp_direction))*yp_direction)/(yp_mat1+yp_mat2)

            yzp_mat1 = np.matrix(np.diag(Zeros_G))*np.matrix(np.ones([len(Zeros_G), len(Poles_G)]))
            yzp_mat2 = np.matrix(np.ones([len(Zeros_G), len(Poles_G)]))*np.matrix(np.diag(Poles_G))

            Qzp = np.transpose(np.conjugate(yz_direction))*yp_direction/(yzp_mat1-yzp_mat2)


            #this matrix is the matrix from which the SVD is going to be done to determine the final minimum peak
            pre_mat = (sc_lin.sqrtm((np.linalg.inv(Qz)))*Qzp*(sc_lin.sqrtm(np.linalg.inv(Qp))))

            #final calculation for the peak value
            Ms_min = np.sqrt(1+(np.max(np.linalg.svd(pre_mat)[1]))**2)
            print ''
            print 'Minimum peak values on T and S'
            print 'Ms_min = Mt_min = ', Ms_min
            print ''

        else:
            print ''
            print 'Minimum peak values on T and S'
            print 'No limits on minimum peak values'
            print ''

    #check for dead time
    #dead_G = deadtime[0]
    #dead_gd = deadtime[1]

    #if np.sum(dead_G)!= 0:
        #therefore deadtime is present in the system therefore extra precautions need to be taken
        #manually set up the dead time matrix

    #    dead_m = np.zeros([len(Poles_G), len(Poles_G)])


    #    for i in range(len(Poles_G)):
    #        for j in range(len(Poles_G))
    #            dead_m

    #plant with RHP zeros from 6-48
    #checking that the plant and controlled variables have the ability to reject load disturbances




    #eq 6-50 pg 240 from skogestad
    #eg 6-50 pg 240 from skogestad for simultanious disturbacne matrix
    #Checking input saturation for perfect control for disturbance rejection
    #checking for maximum disturbance just at steady state

    [U_gd, S_gd, V_gd] = np.linalg.svd(Gd(0.000001))
    y_gd_max = np.max(S_gd)*U_gd[:, 0]
    mod_G_gd_ss = np.max(np.linalg.inv(G(0.000001))*y_gd_max)



    print 'Perfect control input saturation from disturbances'
    print 'Needs to be less than 1 '
    print 'Max Norm method'
    print 'Checking input saturation at steady state'
    print 'This is done by the worse output direction of Gd'
    print mod_G_gd_ss
    print ''

    #
    #
    #

    print 'Figure 1 is for perfect control for simultaneous disturbances'
    print 'All values on each of the graphs should be smaller than 1'
    print ''

    print 'Figure 2 is the plot of G**1 gd'
    print 'The values of this plot needs to be smaller or equal to 1'
    print ''


    w = np.logspace(w_start, w_end, 100)



    mod_G_gd = np.zeros(len(w))
    mod_G_Gd = np.zeros([np.shape(G(0.0001))[0], len(w)])

    for i in range(len(w)):
        [U_gd, S_gd, V_gd] = np.linalg.svd(Gd(1j*w[i]))
        gd_m = np.max(S_gd)*U_gd[:, 0]
        mod_G_gd[i] = np.max(np.linalg.pinv(G(1j*w[i]))*gd_m)

        mat_G_Gd = np.linalg.pinv(G(w[i]))*Gd(w[i])
        for j in range(np.shape(mat_G_Gd)[0]):
            mod_G_Gd[j, i] = np.max(mat_G_Gd[j, :])

    #def for subplotting all the possible variations of mod_G_Gd


    plot_direction(np.ones([2, len(w)]), 'Perfect control Gd', 'r', 1)
    plot_direction(mod_G_Gd, 'Perfect control Gd', 'b', 1)

    plt.figure(2)
    plt.title('Input Saturation for perfect control |inv(G)*gd|<= 1')
    plt.xlabel('w')
    plt.ylabel('|inv(G)* gd|')
    plt.semilogx(w, mod_G_gd)
    plt.semilogx([w[0], w[-1]], [1, 1])
    plt.semilogx(w[0], 1.1)

    #def G_gd(w):
    #    [U_gd, S_gd, V_gd] = np.linalg.svd(Gd(1j*w))
    #    gd_m = U_gd[:, 0]
    #    mod_G_gd[i] = np.max(np.linalg.inv(G(1j*w))*gd_m)-1
    #    return mod_G_gd

    #w_mod_G_gd_1 = sc_opt.fsolve(G_gd, 0.001)


    #print 'frequencies till which input saturation would not occurs'
    #print w_mod_G_gd_1

    #eq 6-43 pg 238 disturbance condition number
    #this in done over a frequency range to see if possible problems at higher frequencies


    #checking the disturbance condition number at steady  state



    #eq 6-48 pg 239 for plant with RHP zeros
    #



    #
    #
    #

    print 'Figure 3 is the singular value of an specific output with input and disturbance direction vector'
    print 'The solid blue line needs to be large than the red line'
    print 'This only needs to be checked up to frequencies where |u**H gd| >1'
    print ''

    #checking input saturation for acceptable control  disturbance rejection
    #equation 6-55 pg 241 in skogestad
    #checking each singular values and the associated input vector with output direction vector of Gd
    #just for square systems for know

    #revised method including all the possibilities of outputs i
    store_rhs_eq = np.zeros([np.shape(G(0.0001))[0], len(w)])
    store_lhs_eq = np.zeros([np.shape(G(0.0001))[0], len(w)])

    for i in range(len(w)):
        for j in range(np.shape(G(0.0001))[0]):
            store_rhs_eq[j, i] = np.abs(np.transpose(np.conjugate(np.linalg.svd(G(w[i]))[2][:, j]))*np.max(np.linalg.svd(Gd(w[i]))[1])*np.linalg.svd(Gd(w[i]))[0][:, 0])-1
            store_lhs_eq[j, i] = sc_lin.svd(G(w[i]))[1][j]

    plot_direction(store_rhs_eq, 'Acceptable control eq6-55', 'r', 3)
    plot_direction(store_lhs_eq, 'Acceptable control eq6-55', 'b', 3)

    #
    #
    #

    print 'Figure 4 is to check input saturation for reference changes'
    print 'Red line in both graphs needs to be larger than the blue line for values w < wr'
    print 'Shows the wr up to where control is needed'
    print ''

    #checking input saturation for perfect control with reference change
    #eq 6-52 pg 241

    input_sat_reference = [np.min(np.linalg.svd(np.linalg.pinv(reference_change())*G(w_i))[1]) for w_i in w]

    #checking input saturation for perfect control with reference change
    #another equation for checking input saturation with reference change
    #eq 6-53 pg 241

    singular_min_G_ref_track = [np.min(np.linalg.svd(G(1j*w_i))[1]) for w_i in w]

    plt.figure(4)
    plt.subplot(211)
    plt.title('min_sing(G(jw)) minimum requirement')
    plt.loglog(w, singular_min_G_ref_track, 'r')
    plt.loglog([w[0], w[-1]], [1, 1], 'b')
    plt.loglog(w[0], 1.2)
    plt.loglog(w[0], 0.8)
    plt.loglog([wr, wr], [np.min([0.8, np.min(singular_min_G_ref_track)]), np.max([1.2, np.max(singular_min_G_ref_track)])])

    plt.subplot(212)
    plt.title('min_sing(inv(R)*G(jw)) combined changes')
    plt.loglog(w, input_sat_reference, 'r')
    plt.loglog([w[0], w[-1]], [1, 1], 'b')
    plt.loglog(w[0], 1.2)
    plt.loglog(w[0], 0.8)
    plt.loglog([wr, wr], [np.min([0.8, np.min(input_sat_reference)]), np.max([1.2, np.max(input_sat_reference)])])

    #
    #
    #

    print 'Figure 5 is the maximum and minimum singular values of G over a frequency range'
    print 'Figure 5 is also the maximum and minimum singular values of Gd over a frequency range'
    print 'Blue is the minimum values and Red is the maximum singular values'
    print 'Plot of Gd should be smaller than 1 else control is needed at frequencies where Gd is bigger than 1'
    print ''

    #checking input saturation for acceptable control with reference change
    #added check for controllability is the minimum and maximum singular values of system transfer function matrix
    # as a function of frequency
    #condition number added to check for how prone the system would be to uncertainty

    singular_min_G = [np.min(np.linalg.svd(G(1j*w_i))[1]) for w_i in w]
    singular_max_G = [np.max(np.linalg.svd(G(1j*w_i))[1]) for w_i in w]
    singular_min_Gd = [np.min(np.linalg.svd(Gd(1j*w_i))[1]) for w_i in w]
    singular_max_Gd = [np.max(np.linalg.svd(Gd(1j*w_i))[1]) for w_i in w]
    condition_num_G = [np.max(np.linalg.svd(G(1j*w_i))[1])/np.min(np.linalg.svd(G(1j*w_i))[1]) for w_i in w]

    plt.figure(5)
    plt.subplot(311)
    plt.title('min_S(G(jw)) and max_S(G(jw))')
    plt.loglog(w, singular_min_G, 'b')
    plt.loglog(w, singular_max_G, 'r')

    plt.subplot(312)
    plt.title('Condition number of G')
    plt.loglog(w, condition_num_G)

    plt.subplot(313)
    plt.title('min_S(Gd(jw)) and max_S(Gd(jw))')
    plt.loglog(w, singular_min_Gd, 'b')
    plt.loglog(w, singular_max_Gd, 'r')
    plt.loglog([w[0], w[-1]], [1, 1])


    plt.show()

    return Ms_min


PEAK_MIMO(-4, 5, 0.00001, 0.1)
