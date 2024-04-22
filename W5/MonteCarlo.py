import numpy as np
import scipy.constants
import time
from scipy import special as sp
from scipy.interpolate import interp1d
import os

epsi = 1
sigma = 1
lj_potential = lambda r: 4 * epsi * ((sigma / r) ** 12 - (sigma / r) ** 6)

# Function to calculate distance between two points with periodic boundary conditions
def distance(r1, r2):
    dx = r1[0] - r2[0]
    dy = r1[1] - r2[1]
    dx -= np.round(dx / L) * L
    dy -= np.round(dy / L) * L
    return np.sqrt(dx**2 + dy**2)

def TotEnergy(positions, L, N, T):
    E = scipy.constants.Boltzmann * T
    for i in range(N):
        for j in range(i+1, N):
            r = distance(positions[i], positions[j])
            E += lj_potential(r)
    return E

# Compute radial distribution function g(r)
def calculate_radial_distribution(positions, dr, rho, L):
    N = len(positions)
    max_bin_index = int(L / 2 / dr)
    g_values = np.zeros(max_bin_index)

    for i in range(N):
        for j in range(i+1, N):
            r = distance(positions[i], positions[j])
            if r < L / 2:
                bin_index = min(int(r / dr), max_bin_index - 1)
                g_values[bin_index] += 2  # Count each pair only once

    # Normalize g(r)
    for i in range(len(g_values)):
        r_lower = i * dr
        r_upper = (i + 1) * dr
        shell_volume = np.pi * ((r_upper)**2 - (r_lower)**2)
        g_values[i] /= shell_volume * rho * N

    r_values = np.arange(dr, (max_bin_index + 1) * dr, dr)
    return r_values, g_values

def calculate_radial_distribution2(positions, dr, rho, L):
    hist, bins = np.histogramdd(positions, bins=np.linspace(0, L, 50))
    r = bins[:-1, 0] + bins[1:, 0]  # Bin centers
    dr = bins[1, 0] - bins[0, 0]
    n_atoms = len(positions)
    volume = L**2
    rdf = hist.flatten() / (n_atoms * dr * volume)
    return r, rdf

## PARAMETERS ##
N = 242  # Number of atoms
T = 0.5 # Temperature
p = 0.96  # Density
L = np.sqrt(N / p)  # Box size
D = 0.15  # Displacement amplitude
n_steps = 1000

# Initialize positions randomly
positions = np.random.rand(N, 2) * L

# Initialize energy
energy = 0.0

Tstart = time.time()

for step in range(n_steps):
    # Randomly select an atom
    atom_index = np.random.randint(N)
    
    # Calculate energy before displacement
    old_energy = scipy.constants.Boltzmann * T
    for i in range(N):
        if i != atom_index:
            old_energy += lj_potential(distance(positions[atom_index], positions[i]))
    
    # Propose a random displacement
    displacement = (np.random.rand(2) - 0.5) * D
    new_position = positions[atom_index] + displacement
    if np.any(new_position < 0) or np.any(new_position >= L):
        continue
    
    # Apply periodic boundary conditions
    new_position = np.mod(new_position, L)
    
    # Calculate energy after displacement
    new_energy = scipy.constants.Boltzmann * T
    for i in range(N):
        if i != atom_index:
            new_energy += lj_potential(distance(new_position, positions[i]))
    
    # Metropolis criterion
    delta_energy = new_energy - old_energy
    if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / T):
        positions[atom_index] = new_position

print("Tempo di esecuzione:", time.time() - Tstart)

#Salva su un unico file
np.savetxt("configurazione_" + time.strftime("%Y%m%d-%H%M%S") + ".txt", positions)
np.savetxt("parametri_" + time.strftime("%Y%m%d-%H%M%S") + ".txt", [N, T, p, D, n_steps, epsi, sigma, time.time() - Tstart])