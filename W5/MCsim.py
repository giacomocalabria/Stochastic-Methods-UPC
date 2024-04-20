import math
from matplotlib.backend_tools import add_tools_to_manager
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import scipy.constants
N = 242 # number of atoms
L = 10.0 # box size

T = 0.5 # temperature
rho = 0.96 # density
sigma = 3.405 # LJ sigma

def LJpotential(r):
    sigma = 3.405
    epsi = 119.8 * scipy.constants.Boltzmann
    return 4 * epsi * ((sigma / r) ** 12 - (sigma / r) ** 6)

def total_energy(T,x,y):
    E = 0
    E += 1/2*scipy.constants.Boltzmann*T

    for i in range(N):
        for j in range(i+1,N):
            r = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            E += LJpotential(r)