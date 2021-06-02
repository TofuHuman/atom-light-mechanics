## Attempts integrating single gravity-assisted Cs atom trajectories released 
## from a MOT cloud for different update algorithm steps: Euler, Verlet and RK 
## under red-detuned Gaussian dipole field and blue-detuned hollow beams

import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors

########################################################################################################
######################################## UNITS and CONSTANTS ###########################################

# distance: 1 unit is 1 micrometer;
# time: 1 unit is 1 microsecond;
c = 2.9979e8               # speed of light (mu-m/mu-s)
wvlngth = 0.894597447 
                        #0.935            # wavelength of trap laser (mu-m)
                        #sensitive to wavelength
omega = 2*np.pi*c/wvlngth  # angular frequency of trap laser 
eps0 = 8.8541878128e-12    # vacuum permitivity 

g = 9.805e-6        # gravitational constant in (mu-m)/(mu-s)^2
T  = 0.000032       # temperature = 30 mu K
kB = 1.38065e-23    # boltzmann constant in (mu-m)^2/(mu-s)^2 kg/K  units

# caesium-atom-properties
m  = 2.2069e-25                      # mass of caesium atom in kg
wD1 = 2*np.pi*c/(894.59295986e-3)    # angular frequency of D1 line (1/(mu-s))
wD2 = 2*np.pi*c/(852.34727582e-3)    # angular frequency of D2 line (1/(mu-s))
gammaD1 = 2*np.pi*4.5612             # natural linewidth of D1 (1/(mu-s))
gammaD2 = 2*np.pi*5.2227             # natural linewidth of D2 (1/(mu-s))

# parameter for conservative dipole trap force
alphaD1 = -1.5*np.pi*(c**2)*(wD1**(-3))*(gammaD1/(wD1-omega)+gammaD1/(wD1+omega)) # U_D1 = alphaD1 * I;
alphaD2 = -1.5*np.pi*(c**2)*(wD2**(-3))*(gammaD2/(wD2-omega)+gammaD2/(wD2+omega)) # U_D2 = alphaD2 * I;
alpha = -((1./3.)*alphaD1 + (2./3.)*alphaD2)  # U = - alpha * I; F = alpha * (dI/dr)

# for wavelength = 935 nm:
# alpha = 9.4657e-18

r_hcpf = 3.75
w0 = 2.75                 # width of gaussian-mode in hollow-core (mu-m)
zR = np.pi*w0**2/wvlngth  # rayleigh length (mu-m)
print('zR', zR)

# parameter for dissipative scattering force
# gamma_scattering = beta*I(r)
# source: PRApplied 10, 0440 (2018)
# URL: https://journals.aps.org/prapplied/pdf/10.1103/PhysRevApplied.10.044034
# formula in paper is possibly wrong

hbar = 6.634e-28/(2*np.pi)                       # Plank's constant in kg (mu-m)^2/(mu-s) units
beta = np.pi*c**2/(2*hbar*omega**3)*(gammaD1**2/(omega - wD1)**2 + 2*gammaD2**2/(omega - wD2)**2) 
print('beta', beta)

print('alpha', alpha)

L = 20000 # 20 mm is the fiber length

##########################################################################################################
################################# CONSERVATIVE FORCE FIELD DEFINITIONS ###################################

# only position-dependent optical forces
global use_HOM, use_axicon, use_gaussian

use_HOM = 0
use_axicon = 0
use_gaussian = 1

# intensity of red-detuned gaussian beam
P  = 0.5*1e-7             # laser power (kg (mu-m)^2/(mu-s)^3); 100 mW corresponds to 1e-7
I0 = (2*P)/(np.pi*w0**2)     # intensity
print(I0)

# Effective k_vectors

# n_00 = 0.994
# n_10 = 0.986

n_00 = 0.9982
n_10 = 0.9955

k = 2*np.pi/wvlngth
k_00 = 2*n_00*np.pi/wvlngth
k_10 = 2*n_10*np.pi/wvlngth

# power of the fields 
P_00 = 0.9*P
P_10 = 0.1*P
E_00_amp = np.sqrt(2/(c*eps0)*(2*P_00/(np.pi*w0**2)))
E_10_amp = np.sqrt(2/(c*eps0)*(P_10/(np.pi*w0**2)))
 
print(E_00_amp)
print(E_10_amp)
print(0.5*c*eps0*np.abs(E_00_amp)**2/I0)

# intensity of red-detuned gaussian beam
def red_detuned_dipole_beam_parameters(x, y, z):

    if (z > 0):
        w = w0*np.sqrt(1 + (z/zR)**2)
        wdz = w0*(z/zR)/(np.sqrt(1 + (z/zR)**2))
        gaussian = np.exp(-(2*(x**2 + y**2))/(w**2))
        intensity = I0*(w0/w)**2*gaussian

    else:
        w = w0*np.sqrt(1 + (z/zR)**2)
        wdz = w0*(z/zR)/(np.sqrt(1 + (z/zR)**2))
        gaussian = np.exp(-(2*(x**2 + y**2))/(w**2))
        intensity = (I0*(w0/w)**2*gaussian)

    # else:
    #     w = w0
    #     wdz = 0
    #     gaussian = np.exp(-(2*(x**2 + y**2))/(w**2))
    #     intensity = I0*(w0/w)**2*gaussian
    
    return w, intensity


# intensity of red-detuned gaussian beam with one higher order mode
def red_detuned_dipole_beam_parameters_HOM(x, y, z):
    
    r = np.sqrt(x**2 + y**2)
    H_0 = 1
    
    if (z > 0):
             
        w = w0*(np.sqrt(1 + (z/zR)**2))
        H_1 = (2*np.sqrt(2)*x/w)
        R = z*(1 + (zR/z)**2) 
        gouy_phase = np.arctan(z/zR)
        #phase_factor = (k*z + k*r**2/(2*R) - gouy_phase)
        phase_factor = (k*z - gouy_phase) #(k*z + k*r**2/(2*R) - gouy_phase)
        E_G = E_00_amp*(w0/w)*np.exp(-r**2/w**2)*np.exp(1j*phase_factor + 1j*k_00*L)
        E_HG = E_10_amp/E_00_amp*E_G*H_1*H_0*np.exp(-1j*gouy_phase + 1j*k_10*L)
        
        E = E_G + E_HG 
        #intensity = 0.5*c*eps0*(np.abs(E_G)**2 + np.abs(E_HG)**2 + 2*E_00_amp*E_10_amp*(w0/w)**2*np.exp(-2*r**2/w**2)*H_1*np.cos(gouy_phase)) #E_G*np.conjugate(E_HG) + E_HG*np.conjugate(E_G)) #2*np.cos(gouy_phase_extra)*np.abs(E_G)*np.abs(E_HG))
        intensity = 0.5*c*eps0*(np.abs(E))**2

    else: 

        w = w0
        H_1 = (2*np.sqrt(2)*x/w)
        E_G = E_00_amp*w0/w*np.exp(-r**2/w**2)*np.exp(+1j*k_00*z + 1j*k_00*L)
        E_HG = E_10_amp/E_00_amp*E_G*H_1*H_0*np.exp(+1j*k_10*z + 1j*k_10*L)
              
        E = E_G + E_HG 
        intensity = 0.5*c*eps0*(np.abs(E))**2


    return w, intensity


# intensity blue-detuned hollow-beam
global m_r_axicon, m_U_axicon, m_w_axicon

# linear fit parameters for axicon beam
# (Appendix D.4): https://projects.iq.harvard.edu/files/lukin/files/michal_bajcsy_thesis.pdf

m_r_axicon = 0.02264150943
m_U_axicon = -0.11698113207*(kB*1e-6)
m_w_axicon = 0.00793650793

# at fiber-face
r_axicon_0 = 197.35849057
U_axicon_0 = 916.98113207*(kB*1e-6)
w_axicon_0 = 12.06349207

def blue_detuned_axicon_beam_parameters(x, y, z):

    r = np.sqrt(x**2 + y**2)

    r_axicon = (r_axicon_0 + m_r_axicon*z)
    w_axicon = (w_axicon_0 + m_w_axicon*z)
    U_axicon = (U_axicon_0 + m_U_axicon*z)

    axicon_gaussian = np.exp(-(2*(r - r_axicon)**2)/(w_axicon**2))
    axicon_potential = (U_axicon*axicon_gaussian)

    return r_axicon, w_axicon, U_axicon, axicon_gaussian, axicon_potential

# net optical potnetial due to red-detuned, blue-detuned and HOM beams
def net_optical_potential(x, y, z):

    w, intensity = red_detuned_dipole_beam_parameters(x, y, z)
    #w, intensity = red_detuned_dipole_beam_parameters_HOM(x, y, z)
    r_axicon, w_axicon, U_axicon, axicon_gaussian, axicon_potential = blue_detuned_axicon_beam_parameters(x, y, z)
    potential = -1*alpha*intensity + use_axicon*axicon_potential

    return potential

# net optical intensity due to red-detuned and HOM beams
def net_optical_intensity(x, y, z):

    w, intensity = red_detuned_dipole_beam_parameters(x, y, z)
    #r_axicon, w_axicon, U_axicon, axicon_gaussian, axicon_potential = blue_detuned_axicon_beam_parameters(x, y, z)
    #potential = -1*alpha*intensity + use_axicon*axicon_potential

    return intensity

# accelerations analytical expressions of the gradient of a red-detuned gaussian beam
def a_dipole(position):

    x = position[0]
    y = position[1]
    z = position[2]

    r = abs(np.sqrt(x**2 + y**2))

    ax_blue_detuned_hollow = 0
    ay_blue_detuned_hollow = 0
    az_blue_detuned_hollow = 0

    #--------------------------------------#
    # acceleration due to red-detuned gaussian beam
    w, intensity = red_detuned_dipole_beam_parameters(x, y, z)
    ax_red_detuned_gaussian = -4*x*alpha*intensity/(m*w**2)
    ay_red_detuned_gaussian = -4*y*alpha*intensity/(m*w**2)

    if (z>0):
        az_red_detuned_gaussian = alpha*intensity*(4*(z/(zR**2))*(w0/w)**4*((x**2 + y**2)/w0**2) - (2*z/(zR**2))*(w0/w)**2)/m

    else:
        az_red_detuned_gaussian = 0

    #--------------------------------------#
    # acceleration due to blue-detuned hollow beam
    r_axicon, w_axicon, U_axicon, axicon_gaussian, axicon_potential = blue_detuned_axicon_beam_parameters(x, y, z)
    
    # dU/dx = dU/dr*cos(theta)
    ax_blue_detuned_hollow = 4*(r - r_axicon)*axicon_potential/(m*w_axicon**2)*(x/r)
    ay_blue_detuned_hollow = 4*(r - r_axicon)*axicon_potential/(m*w_axicon**2)*(y/r)
    az_blue_detuned_hollow = -1/m*(m_U_axicon*axicon_gaussian + U_axicon*axicon_gaussian*(4*(r - r_axicon)/w_axicon**2*(m_r_axicon) +  4*(r - r_axicon)**2*m_w_axicon/(w_axicon)**3))

    #--------------------------------------#
    # acceleration due to red-detuned higher order mode

    w = w0*(np.sqrt(1 + (z/zR)**2))
    gaussian = np.exp(-(2*(x**2 + y**2))/(w**2))
    intensity_00 = (0.5*c*eps0*np.abs(E_00_amp)**2)*(w0/w)**2*gaussian
    
    ax_red_detuned_gaussian_00 = -4*x*alpha*intensity_00/(m*w**2)
    ay_red_detuned_gaussian_00 = -4*y*alpha*intensity_00/(m*w**2)

    if (z>0):
        az_red_detuned_gaussian_00 = alpha*intensity_00*(4*(z/(zR**2))*(w0/w)**4*((x**2 + y**2)/w0**2) - (2*z/(zR**2))*(w0/w)**2)/m
    else:
        az_red_detuned_gaussian_00 = 0

    I1_factor = np.abs(E_10_amp)**2/np.abs(E_00_amp)**2
    factor_intensity_HG = I1_factor*(8*x**2/(w**2))

    ax_hermite_gaussian_dipole = factor_intensity_HG*ax_red_detuned_gaussian_00 + I1_factor*16*x/(w**2)*intensity_00*alpha/m
    ay_hermite_gaussian_dipole = factor_intensity_HG*ay_red_detuned_gaussian_00
    az_hermite_gaussian_dipole = factor_intensity_HG*az_red_detuned_gaussian_00

    I10_factor = 2*E_10_amp/E_00_amp
    factor_cross_term_x = (2*np.sqrt(2)*x/w)

    if (z > 0):
        factor_cross_term_z = np.cos(np.arctan(z/zR) + k_00*L - k_10*L)
        factor_cross_term_zz = -1/zR*np.sin(np.arctan(z/zR) + k_00*L - k_10*L)*(1/(1 + (z/zR)**2)) 
    else: 
        factor_cross_term_z = np.cos(k_00*z - k_10*z + k_00*L - k_10*L)
        factor_cross_term_zz = -1*np.sin(k_00*z - k_10*z + k_00*L - k_10*L)*(k_00 - k_10) 

    ax_cross_term_dipole = I10_factor*(ax_red_detuned_gaussian_00*factor_cross_term_x*factor_cross_term_z + intensity_00*2*np.sqrt(2)/w*factor_cross_term_z*alpha/m)
    ay_cross_term_dipole = I10_factor*(ay_red_detuned_gaussian_00*factor_cross_term_x*factor_cross_term_z)
    az_cross_term_dipole = I10_factor*(az_red_detuned_gaussian_00*factor_cross_term_x*factor_cross_term_z + intensity_00*factor_cross_term_x*factor_cross_term_zz*alpha/m)

    #--------------------------------------#
    # total acceleration 
    ax = use_gaussian*ax_red_detuned_gaussian + use_axicon*ax_blue_detuned_hollow + use_HOM*(ax_red_detuned_gaussian_00 + ax_hermite_gaussian_dipole + ax_cross_term_dipole)
    ay = use_gaussian*ay_red_detuned_gaussian + use_axicon*ay_blue_detuned_hollow + use_HOM*(ay_red_detuned_gaussian_00 + ay_hermite_gaussian_dipole + ay_cross_term_dipole)
    az = -g + use_gaussian*az_red_detuned_gaussian + use_axicon*az_blue_detuned_hollow + use_HOM*(az_red_detuned_gaussian_00 + az_hermite_gaussian_dipole + az_cross_term_dipole)

    return ax, ay, az


##########################################################################################################
######################################### DISSIPATIVE FORCE ##############################################

global use_momentum_kick, use_friction, N_scattering

use_momentum_kick = 1
use_friction = 0

abs_v_factor = hbar*omega/(m*c)
print('abs_v_factor', abs_v_factor)

# momentum kick
def scattering_momentum_kick_0(position, velocity, N_scattering):

    x = position[0]
    y = position[1]
    z = position[2] 

    I = net_optical_intensity(x, y, z)

    if (int(N_scattering + I*beta*dt) - int(N_scattering) == 1):
        
        theta = np.pi*np.random.uniform(0, 1, 1)
        phi = 2*np.pi*np.random.uniform(0, 1, 1)

        # spontaneous emission factor
        spont_v_factor = hbar*wD1/(m*c)

        # absorption factor 
        abs_v_factor = hbar*omega/(m*c)

        velocity[0] = velocity[0] + spont_v_factor*np.cos(phi)*np.sin(theta) 
        velocity[1] = velocity[1] + spont_v_factor*np.sin(phi)*np.sin(theta)
        velocity[2] = velocity[2] + abs_v_factor + spont_v_factor*np.cos(theta)

        print("Momentum Kick! No.", N_scattering + 1)

    N_scattering = N_scattering + I*beta*dt

    return N_scattering


def scattering_momentum_kick_1(position, velocity, N_scattering):

    x = position[0]
    y = position[1]
    z = position[2] 

    I = net_optical_intensity(x, y, z)
    p_scattering = np.random.uniform(0, 1, 1)

    if (p_scattering < I*beta*dt):

        print('Scattering Probability', I*beta*dt)
        
        theta = np.pi*np.random.uniform(0, 1, 1)
        phi = 2*np.pi*np.random.uniform(0, 1, 1)

        # spontaneous emission factor
        spont_v_factor = hbar*wD1/(m*c)

        # absorption factor 
        abs_v_factor = hbar*omega/(m*c)

        velocity[0] = velocity[0] + spont_v_factor*np.cos(phi)*np.sin(theta) 
        velocity[1] = velocity[1] + spont_v_factor*np.sin(phi)*np.sin(theta)
        velocity[2] = velocity[2] + abs_v_factor + spont_v_factor*np.cos(theta)

        # if ()
        # print("Momentum Kick! No.", N_scattering + 1)

        N_scattering = N_scattering + 1

    return N_scattering

# continuous damping/friction


##########################################################################################################
######################################### PARTICLE INITIALIZATION ########################################

def initialize_particles(Delta_t):

    global N, p, v, a, t, dt, plist, texit, state, E_initial

    N = 1                               # only one particle 

    # particle attributes 
    p = np.zeros((N,3))
    v = np.zeros((N,3))
    a = np.zeros((N,3))
    t = np.zeros(N)

    cz = 5000                                 # height from which particle is dropped

    
    texit = np.zeros(N)                 # keep a tab on exit-time of the fiber 


    state = np.zeros(N)      # state of atom:  0 - not loaded, running
                             #                 1 - not loaded, stopped
                             #                 2 - loaded, running
                             #                 3 - loaded, stopped

    

    #########################################################################################
    ###################### testing collection, red-detuned gaussian #########################
    #########################################################################################

    # # sample loaded particles
    # p[0, 0] = 50
    # p[0, 1] = 50
    # p[0, 2] = 5000
    # v[0, 0] = -0.0015
    # v[0, 1] = 0
    # v[0, 2] = 0

    # test (curious case of not taking care of the sign of az acceleration)
    #      (this particle gets trapped in air, if the vertical acceleration)
    #      (counters gravity)

    # actual gaussian accelerations: (ax, ay = -ve; az = +ve) (positive alpha definition)

    # (trap for (coefficients) ax, ay = -ve; az = +ve..)

    # p[0, 0] = -6.303830
    # p[0, 1] = -10.497642
    # p[0, 2] = 96.674043
    # v[0, 0] = 0.049196
    # v[0, 1] = -0.026824
    # v[0, 2] = -0.017868

    # (trap for (coefficients) ax, ay = -ve; az = -ve..)

    # p[0, 0] = 5.748023
    # p[0, 1] = -9.619855
    # p[0, 2] = 277.464429
    # v[0, 0] = -0.031001
    # v[0, 1] = -0.010055
    # v[0, 2] = -0.048044

    # p[0, 0] = 2.957306
    # p[0, 1] = 11.589821
    # p[0, 2] = 169.793773
    # v[0, 0] = -0.057277
    # v[0, 1] = 0.007341
    # v[0, 2] = -0.003692

    # p[0, 0] = 9.819227
    # p[0, 1] = -10.185924
    # p[0, 2] = 357.078815
    # v[0, 0] = -0.018156
    # v[0, 1] = -0.001412
    # v[0, 2] = 0.026160

    # p[0, 0] = -0.403617
    # p[0, 1] = -6.572135
    # p[0, 2] = 98.675204
    # v[0, 0] = -0.054168
    # v[0, 1]  = 0.000056
    # v[0, 2] = -0.026538


    #########################################################################################
    ###################### testing collection, blue-detuned axicon ##########################
    #########################################################################################

    # p[0, 0] = -50
    # p[0, 1] = 0
    # p[0, 2] = 4650
    # v[0, 0] = 0.04
    # v[0, 1] = 0
    # v[0, 2] = 0.03
    # Temp_check = 0.5*m*(v[0, 0]**2 + v[0, 1]**2)*1e6/(kB)
    # print('First particle KE:', Temp_check)

    # p[1, 0] = 100
    # p[1, 1] = 0
    # p[1, 2] = 4200
    # v[1, 0] = 0.25
    # v[1, 1] = 0
    # v[1, 2] = -0.01
    # Temp_check = 0.5*m*(v[1, 0]**2 + v[1, 1]**2)*1e6/(kB)
    # print('Second particle KE:', Temp_check)

    # p[2, 0] = -600
    # p[2, 1] = 0
    # p[2, 2] = 3500
    # v[2, 0] = 0.02
    # v[2, 1] = 0
    # v[2, 2] = 0.01
    # Temp_check = 0.5*m*(v[2, 0]**2 + v[2, 1]**2)*1e6/(kB)
    # print('Third particle KE:', Temp_check)

    # p[3, 0] = -600
    # p[3, 1] = 0
    # p[3, 2] = 4000
    # v[3, 0] = 0.3
    # v[3, 1] = 0
    # v[3, 2] = -0.1
    # Temp_check = 0.5*m*(v[3, 0]**2 + v[3, 1]**2)*1e6/(kB)
    # print('Fourth particle KE:', Temp_check)



    #########################################################################################
    ###################### testing collection, scattering force #############################
    #########################################################################################

    p[0, 0] = -0.44539460360695243
    p[0, 1] = -1.1485951336856262
    p[0, 2] = 99.99346097804897
    v[0, 0] = 0.054878451912033037
    v[0, 1] = -0.13002106911058417
    v[0, 2] = -0.3167495122563623


    #########################################################################################
    ################################### INITIALIZATIONS #####################################
    #########################################################################################

    # energy = optical potential + gravitational potential + kinetic
    global index_check
    index_check = 0

    GE_initial = m*g*p[index_check,2]
    KE_initial = 0.5*m*(v[index_check,0]**2 + v[index_check,1]**2 + v[index_check,2]**2)
    U_initial = net_optical_potential(p[index_check,0], p[index_check, 1], p[index_check, 2])

    E_initial = KE_initial + U_initial + GE_initial 

    print('KE Initial', KE_initial)
    print('GE Initial', GE_initial)
    print('U Initial', U_initial)
    print('E Initial', E_initial)

    # RK4 and CP parameters and temp. memory allocation
    
    global k1p, k1v, k2p, k2v, k3p, k3v, k4p, k4v, k5p, k5v, k6p, k6v
    global dt_adaptive, dp, dp_1, p_1, A, B, C, C_1

    k1p = np.zeros(3)
    k1v = np.zeros(3)
    k2p = np.zeros(3)
    k2v = np.zeros(3)
    k3p = np.zeros(3)
    k3v = np.zeros(3)
    k4p = np.zeros(3)
    k4v = np.zeros(3)
    k5p = np.zeros(3)
    k5v = np.zeros(3)
    k6p = np.zeros(3)
    k6v = np.zeros(3)

    dp = np.zeros(3)
    dp_1 = np.zeros(3)
    p_1 = np.zeros(3)
    dt_adaptive = Delta_t*np.ones(N)

    # coefficients in cash-karp RK algorithm 
    A = [1./5., 3./10., 3./5., 1., 7./8.]                   # time-dependent forces
    B = [[0, 0, 0, 0, 0], 
         [1./5., 0, 0, 0, 0],
         [3./40., 9./40., 0, 0, 0],
         [3./10., -9./10., 6./5., 0, 0],
         [-11./54., 5./2., -70./27., 35./27., 0],
         [1631./55296., 175./512., 575./13824., 44275./110592., 253./4096.]]
    C = [37./378., 0, 250./621., 125./594., 0, 512./1771.]
    C_1 = [2825./27648., 0, 18575./48384., 13525./55296., 277./14336., 1./4.]

    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    C_1 = np.asarray(C_1)


##########################################################################################################
####################################### UPDATE ALGORITHM FUNCTIONS #######################################

# euler and v-verlet are symplectic
# error is of first and third order in time-step

def euler_update(dt, t, p, v, a):
    t = t + dt
    p[:] = p[:] + v[:]*dt 
    v[:] = v[:] + a[:]*dt
    a[:] = a_dipole(p[:])

    return t, p, v, a

def velocity_verlet_update(dt, t, p, v, a):
    t = t + dt
    v[:] = v[:] + a[:]*dt/2
    p[:] = p[:] + v[:]*dt 
    a[:] = a_dipole(p[:])
    v[:] = v[:] + a[:]*dt/2
    
    return t, p, v, a

# RK4 is non-symplectic     
# error is fourth order in time-step

def RK4_update(dt, t, p, v, a):
    t = t + dt 

    k1v[:] = a[:]*dt
    k1p[:] = v[:]*dt
    
    k2v[:] = a_dipole(p[:] + k1p[:]/2)
    k2v[:] = k2v[:]*dt
    k2p[:] = (v[:] + k1v[:]/2)*dt
    
    k3v[:] = a_dipole(p[:] + k2p[:]/2)
    k3v[:] = k3v[:]*dt
    k3p[:] = (v[:] + k2v[:]/2)*dt
    
    k4v[:] = a_dipole(p[:] + k3p[:])
    k4v[:] = k4v[:]*dt
    k4p[:] = (v[:] + k3v[:])*dt

    p[:] = p[:] + 1/6*(k1p[:] + 2*k2p[:] + 2*k3p[:] + k4p[:])
    v[:] = v[:] + 1/6*(k1v[:] + 2*k2v[:] + 2*k3v[:] + k4v[:])
    a[:] = a_dipole(p[:])

    return t, p, v, a


# cash-karp is RK-esque non-symplectic integrators
# it implements an adaptive step-size

# help! 
def RKCK_update(dt, t, p, v, a, tol):

    err = 2*tol

    while (err > tol):

        k1v[:] = a[:]*dt
        k1p[:] = v[:]*dt
    
        k2v[:] = a_dipole(p[:] + B[1,0]*k1p[:])
        k2v[:] = k2v[:]*dt
        k2p[:] = (v[:] + B[1, 0]*k1v[:])*dt

        k3v[:] = a_dipole(p[:] + B[2, 0]*k1p[:] + B[2, 1]*k2p[:])
        k3v[:] = k3v[:]*dt
        k3p[:] = (v[:] + B[2, 0]*k1v[:] + B[2, 1]*k2v[:])*dt

        k4v[:] = a_dipole(p[:] + B[3, 0]*k1p[:] + B[3, 1]*k2p[:] +  B[3, 2]*k3p[:])
        k4v[:] = k4v[:]*dt
        k4p[:] = (v[:] + B[3, 0]*k1v[:] + B[3, 1]*k2v[:] +  B[3, 2]*k3v[:])*dt

        k5v[:] = a_dipole(p[:] + B[4, 0]*k1p[:] + B[4, 1]*k2p[:] +  B[4, 2]*k3p[:] + B[4, 3]*k4p[:])
        k5v[:] = k5v[:]*dt
        k5p[:] = (v[:] + B[4, 0]*k1v[:] + B[4, 1]*k2v[:] +  B[4, 2]*k3v[:] + B[4, 3]*k4v[:])*dt

        k6v[:] = a_dipole(p[:] + B[5, 0]*k1p[:] + B[5, 1]*k2p[:] +  B[5, 2]*k3p[:] + B[5, 3]*k4p[:] + B[5, 4]*k5p[:])
        k6v[:] = k6v[:]*dt
        k6p[:] = (v[:] + B[5, 0]*k1v[:] + B[5, 1]*k2v[:] +  B[5, 2]*k3v[:] + B[5, 3]*k4v[:] + B[5, 4]*k5v[:])*dt

        dp[:] = (C[0]*k1p[:] + C[1]*k2p[:] + C[2]*k3p[:] + C[3]*k4p[:] + C[4]*k5p[:] + C[5]*k6p[:])
        dp_1[:] = (C_1[0]*k1p[:] + C_1[1]*k2p[:] + C_1[2]*k3p[:] + C_1[3]*k4p[:] + C_1[4]*k5p[:] + C_1[5]*k6p[:])

        err =  1e-16 + max(abs(dp - dp_1))
        dt = 0.1 * dt * (tol/err)**(1/5)
    
    p[:] = p[:] + dp[:]
    v[:] = v[:] + (C[0]*k1v[:] + C[1]*k2v[:] + C[2]*k3v[:] + C[3]*k4v[:] + C[4]*k5v[:] + C[5]*k6v[:])
    a[:] = a_dipole(p[:])

    t = t + dt 
    # print('Err', err)
    # print('dt', dt) 

    return dt, t, p, v, a

# others 

def RKF45_update(dt_adaptive, t, p, v, a):
    print('RK-Fehlberg: Wait Dude!')


def BS_update(dt_adaptive, t, p, v, a):
    print('Bulirsch-Stoer: Do we want to do this?')

# PC is a non-starting multi-step alternative    
def PC_update(dt_adaptive, t, p, v, a):
    print('Predictor-Corrector: Do we want to do this?')


##########################################################################################################
################################ TIME and PARTICLE EVOLUTION LOOPS #######################################

def time_particle_looper(Delta_t, solver):

    dt = Delta_t

    if (solver == 0):
        print('Euler Updater with Time-Step:', Delta_t)
    
    if (solver == 1):
        print('Velocity Verlet with Time-Step:', Delta_t)

    if (solver == 2):
        print('RK4 with Time-Step:', Delta_t)

    if (solver == 3):
        print('Cash-Karp with Adaptive Time-Step')
        tol = 0.00000001  #input("Error Tolerance: ")
        #tol = np.float(tol)

    cz = 5000
    dt = Delta_t                                 # fixed time-step
    T_upper = 1000000 #int(2*np.sqrt((cz)/g)*(1/dt))      # some upper bound for transit time-steps
    plist = np.zeros((N,3,T_upper))

    counter = 0
    T = 0                                # iterator
    print('start evolving')
    start = time.time()
    z_stop = -50

    z_above = 101

    N_scattering = 0
    printer = 0
    # time loop
    while (counter < N and T < T_upper):

        # particle loop 
        for i in range(N):

            if (state[i] == 0 or state[i] == 2):

                # if (p[i,2] < 2000): 
                #     dt = 0.01

                # if (p[i,2] < 400): 
                #     dt = 0.001

                # if (p[i,2] < 10): 
                #     dt = 0.0001

                if (p[i,2] < 100 and printer == 0):

                    printer = 1
                    print('x', p[i,0]) 
                    print('y', p[i,1])
                    print('z', p[i,2])
                    print('vx', v[i,0])
                    print('vy', v[i,1])
                    print('vz', v[i,2])

                # particle updator
                if (solver == 0):
                    t[i], p[i,:], v[i,:], a[i,:] = euler_update(dt, t[i], p[i,:], v[i,:], a[i,:])

                if (solver == 1):
                    t[i], p[i,:], v[i,:], a[i,:] = velocity_verlet_update(dt, t[i], p[i,:], v[i,:], a[i,:])
                
                if (solver == 2):
                    t[i], p[i,:], v[i,:], a[i,:] = RK4_update(dt, t[i], p[i,:], v[i,:], a[i,:])
    
                if (solver == 3):
                    dt_adaptive[i], t[i], p[i,:], v[i,:], a[i,:] = RKCK_update(dt_adaptive[i], t[i], p[i,:], v[i,:], a[i,:], tol)
                

                if (use_momentum_kick == 1):
                    N_scattering = scattering_momentum_kick_1(p[i,:], v[i,:], N_scattering)
                    #print(N_scattering)

                # track particle location
                plist[i, :, T] = p[i, :]

                # particle status updates 
                if (p[i,2] < 0 and (p[i,0]**2 + p[i,1]**2) < r_hcpf**2):
                    state[i] = 2
                    #print(p[0,0], p[0,1], p[0,2], v[0,0], v[0,1], v[0,2])

                if (p[i,2] < z_stop and (p[i,0]**2 + p[i,1]**2) < r_hcpf**2):
                    state[i] = 3
                    counter = counter + 1
                    texit[i] = T
                    print('loaded!')
                    continue

                if (p[i,2] > z_above or p[i,2] < z_stop and (p[i,0]**2 + p[i,1]**2) > r_hcpf**2):
                    state[i] = 1
                    counter = counter + 1
                    texit[i] = T
                    print('wasted!')
                    continue

        if (counter == N):
            print('all done')
            break

        T = T + 1

    if (texit[0] == 0):
        texit[0] = T_upper

    end = time.time()
    time_evolve = (end - start)
    print('Time for evolving {} particles:'.format(N), end - start, 's')
    print('Iterations in for loop:', T)

    KE_final = 0.5*m*(v[index_check,0]**2 + v[index_check,1]**2 + v[index_check,2]**2)
    U_final = net_optical_potential(p[index_check,0], p[index_check, 1], p[index_check, 2])
    GE_final = m*g*p[index_check,2]
    E_final = KE_final + GE_final + U_final

    energy_conservation_checker = (E_final - E_initial)/(E_initial)
    print('Energy Initial:', E_initial)
    print('Energy Final:', E_final)
    print('Energy Check:', energy_conservation_checker)

    print('Number of Scattering Events', N_scattering)

    plist_cropped = plist

    return time_evolve, energy_conservation_checker, plist_cropped


##########################################################################################################
############################# PLOTTING: 'STABILITY' AND TRAJECTORIES #####################################

# # # # basic check 
dt = 0.0001
initialize_particles(dt)
time_evolve, energy_conservation_checker, plist_plotter = time_particle_looper(dt, 1)

# plot style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 4
plt.rcParams.update({'figure.autolayout': True})
 
# time-step versus energy conservation plot 
Delta_t_list = [1e0, 0.5, 1e-1, 0.5*1e-1, 1e-2] #, 0.5*1e-2, 1e-3]

def dt_energy_plotter():

    n_delta = len(Delta_t_list)
    n_algo = 3
    energy_conservation_plotter = np.zeros((n_delta, n_algo))
    time_evolve_plotter = np.zeros((n_delta, n_algo))

    colours = ['b^--', 'g^--', 'r^--', 'm']
    plot_labels = ['Euler', 'Velocity-Verlet', 'RK4', 'RKCK - Adaptive $\Delta t$']

    # colours = ['m']
    # plot_labels = ['RKCK - Adaptive $\Delta t$']

    fig_t, ax_t = plt.subplots()
    fig_e, ax_e = plt.subplots()

    for j in range(n_algo):

        for i in range(n_delta):

            initialize_particles(Delta_t_list[i])
            time_evolve_plotter[i, j], energy_conservation_plotter[i, j] = time_particle_looper(Delta_t_list[i], j)
        
        ax_t.plot(Delta_t_list, time_evolve_plotter[:, j], colours[j], label=plot_labels[j])
        ax_t.set_xscale('log')
        ax_t.set_yscale('log')

        ax_e.plot(Delta_t_list, abs(energy_conservation_plotter[:, j]), colours[j], label=plot_labels[j])
        ax_e.set_xscale('log')
        ax_e.set_yscale('log')
    
    ax_t.plot(0, 0, colours[3], label=plot_labels[3])
    ax_e.plot(0, 0, colours[3], label=plot_labels[3])
       
    fig_t.tight_layout()
    ax_t.set_ylabel(r"$T_{evolve}$ (s)")
    ax_t.set_xlabel("$\Delta t$ ($\mu$s)")
    ax_t.legend()
    fig_t.savefig('time_integrator-comparison.png', transparent=True)
        
    fig_e.tight_layout()
    ax_e.set_ylabel(r"$|\frac{\Delta E}{E}|$", fontsize=16)
    ax_e.set_xlabel("$\Delta t$ ($\mu$s)")
    ax_e.legend()
    fig_e.savefig('energy_integrator-comparison.png', transparent=True)

    plt.show()

#dt_energy_plotter()

# 2D plot  

def twoD_plotter():

    # x-z side profile 
    x_range = 20 #2000
    z_top = 100
    z_bottom = -50#-1000
    x = np.arange(-1*x_range, x_range, 0.1)
    y = x
    z_above = np.arange(0.0001, z_top, 0.1)
    z_below = np.arange(z_bottom, -0.0001, 0.1)
    z = np.arange(z_bottom, z_top, 1)

    # dipole intensity map 
    def intensity_map_above(x, z):
        zr = np.pi*w0**2/wvlngth
        wz = w0*np.sqrt(1 + (z/zr)**2)        
        return 1*(w0/wz)**2*np.exp(-2*x**2/(wz**2))

    def intensity_map_below(x, z):
        wz = w0       
        return 1*(w0/wz)**2*np.exp(-2*x**2/(wz**2))

    # axicon intensity map 
    def axicon_map(x, z):
        r_axicon, w_axicon, U_axicon, axicon_gaussian, axicon_potential = blue_detuned_axicon_beam_parameters(x, 0, z)
        return axicon_gaussian

    # HOM intensity map

    def intensity_map_above_HOM(x, z):
        y = 0
        r = np.sqrt(x**2 + y**2)  
        w = w0*(np.sqrt(1 + (z/zR)**2))
        H_1 = (2*np.sqrt(2)*x/(w))
        H_0 = 1

        gouy_phase = np.arctan(z/zR)
    
        w = w0*(np.sqrt(1 + (z/zR)**2))         
        R = z*(1 + (zR/z)**2) 
        phase_factor = (k*z - gouy_phase) #(k*z + k*r**2/(2*R) - gouy_phase)
        E_G = E_00_amp*w0/w*np.exp(-r**2/w**2)*np.exp(1j*phase_factor + 1j*k_00*L)

        gouy_phase_extra = np.arctan(z/zR)
        E_HG = E_10_amp/E_00_amp*E_G*H_1*H_0*np.exp(-1j*gouy_phase_extra + 1j*k_10*L)

        E = E_G + E_HG 
        intensity = np.abs(E**2)

        return intensity

    def intensity_map_below_HOM(x, z):
        y = 0
        r = np.sqrt(x**2 + y**2)  
        H_1 = (2*np.sqrt(2)*x/(w0))
        H_0 = 1

        w = w0
        E_G = E_00_amp*w0/w*np.exp(-r**2/w**2)*np.exp(1j*k_00*z + 1j*k_00*L)
        E_HG = E_10_amp/E_00_amp*E_G*H_1*H_0*np.exp(1j*k_10*z + 1j*k_10*L)

        E = E_G + E_HG 
        intensity = np.abs(E**2)

        return intensity
        
    X, Z_above = np.meshgrid(x, z_above)
    I_dipole_above = intensity_map_above(X, Z_above)

    X, Z_below = np.meshgrid(x, z_below)
    I_dipole_below = intensity_map_above(X, Z_below)

    # X, Z = np.meshgrid(x, z)
    # I_axicon = axicon_map(X, Z)

    # X, Z_above = np.meshgrid(x, z_above)
    # I_HOM_above = intensity_map_above_HOM(X, Z_above)

    # X, Z_below = np.meshgrid(x, z_below)
    # I_HOM_below = intensity_map_below_HOM(X, Z_below)

    # plot field
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    im1 = plt.imshow(I_dipole_above, norm=colors.PowerNorm(gamma=1./7.),
                cmap='Greens', extent=[-1*x_range, x_range, 0, z_top], origin="lower")

    
    im1_a = plt.imshow(I_dipole_below, norm=colors.PowerNorm(gamma=1./7.),
                cmap='Greens', extent=[-1*x_range, x_range, z_bottom, 0], origin="lower")

    
    # im2 = plt.imshow(I_axicon, norm=colors.PowerNorm(gamma=1./7.),
    #             cmap='Blues', extent=[-1*x_range, x_range, z_bottom, z_top], origin="lower", alpha=1)

    # im3 = plt.imshow(I_HOM_above, norm=colors.PowerNorm(gamma=1./7.),
    #              cmap='Greens', extent=[-1*x_range, x_range, 0, z_top], origin="lower")

    
    # im3_a = plt.imshow(I_HOM_below, norm=colors.PowerNorm(gamma=1./7.), cmap='Greens', extent=[-1*x_range, x_range, z_bottom, 0], origin="lower")


    # # plot trajectory 
    plt.plot(plist_plotter[0,0,0:int(texit[0])], plist_plotter[0,2,0:int(texit[0])], 'm', label='Random Trial')
    # plt.plot(plist_plotter[1,0,0:int(texit[1])], plist_plotter[1,2,0:int(texit[1])], 'r', label='Random Trial')
    # plt.plot(plist_plotter[2,0,0:int(texit[2])], plist_plotter[2,2,0:int(texit[2])], 'y', label='Random Trial')
    # plt.plot(plist_plotter[3,0,0:int(texit[3])], plist_plotter[3,2,0:int(texit[3])], 'g', label='Random Trial')

    ax.set_xlim(-1*x_range,  x_range)
    ax.set_ylim(z_bottom, z_top)
    # plt.yticks([0, 1000, 2000, 3000, 4000, 5000], [0, 1, 2, 3, 4, 5])
    # plt.xticks([-600, -250, 0, 250, 600])
    ax.set_xlabel('x $\mu$m')
    ax.set_ylabel('z $\mu$m')
    plt.show()


    # x = np.arange(-1*6, 6, 0.01)
    # y = x
    
    # def intensity_map_above_HOM(x, y):

    #     z = 3
    #     r = np.sqrt(x**2 + y**2)  
    #     w = w0*(np.sqrt(1 + (z/zR)**2))
    #     H_1 = (2*np.sqrt(2)*x/(w))
    #     H_0 = 1

    #     gouy_phase = np.arctan(z/zR)
    #     w = w0*(np.sqrt(1 + (z/zR)**2))         
    #     R = z*(1 + (zR/z)**2) 
    #     phase_factor = (k*z + k*r**2/(2*R) - gouy_phase)

    #     phase_factor = (k*z - gouy_phase) #(k*z + k*r**2/(2*R) - gouy_phase)
    #     E_G = E_00_amp*w0/w*np.exp(-r**2/w**2)*np.exp(-1j*phase_factor)

    #     gouy_phase_extra = np.arctan(z/zR)
    #     E_HG = E_10_amp/E_00_amp*E_G*H_1*H_0*np.exp(1j*gouy_phase_extra)

    #     E = E_HG + E_G 
    #     intensity = np.abs(E**2)

    #     return intensity

    # X, Y = np.meshgrid(x, y)
    # I_HOM_above = intensity_map_above_HOM(X, Y)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # im3 = plt.imshow(I_HOM_above, cmap='Greens', origin="lower")
    # plt.show()


twoD_plotter()

# 3D plot

def threeD_plotter():

    plist_size_arr = np.shape(plist_plotter)
    plist_size = int(plist_size_arr[2])
    plist_sample = plist_size - 500

    def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
        z = np.linspace(0, height_z, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid=np.meshgrid(theta, z)
        x_grid = radius*np.cos(theta_grid) + center_x
        y_grid = radius*np.sin(theta_grid) + center_y
        return x_grid,y_grid,z_grid

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    Xc,Yc,Zc = data_for_cylinder_along_z(0, 0, w0, -10)
    ax.plot_surface(Xc, Yc, Zc, alpha=0.5)
    ax.plot(plist_plotter[0,0,plist_sample:plist_size], plist_plotter[0,1,plist_sample:plist_size], plist_plotter[0,2,plist_sample:plist_size])
    ax.set_xlabel('x $\mu$m')
    ax.set_ylabel('y $\mu$m')
    ax.set_zlabel('z $\mu$m')
    plt.savefig('sample.png', transparent=True)
    plt.show()

#threeD_plotter()

