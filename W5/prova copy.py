import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from tqdm import tqdm  # for progress bar (optional)
import time
from scipy import special as sp
from scipy.interpolate import interp1d

# Lennard-Jones potential
def lj_potential(r):
    sigma = 3.405   # Angstrom (Ar) given in the exercise
    epsi = 119.8 * scipy.constants.Boltzmann
    epsi = 1
    return 4 * epsi * ((sigma / r) ** 12 - (sigma / r) ** 6)

# Function to calculate distance between two points with periodic boundary conditions
def distance(r1, r2, L):
    dx = r1[0] - r2[0]
    dy = r1[1] - r2[1]
    dx -= np.round(dx / L) * L
    dy -= np.round(dy / L) * L
    return np.sqrt(dx**2 + dy**2)

def TotEnergy(positions, L, N):
    E = 0
    for i in range(N):
        for j in range(i+1, N):
            r = distance(positions[i], positions[j], L)
            E += lj_potential(r)
    return E

# Compute radial distribution function g(r)
def calculate_radial_distribution(positions, L, dr=0.05):
    N = len(positions)
    rho = N / (L * L)
    max_bin_index = int(L / 2 / dr)
    g_values = np.zeros(max_bin_index)

    for i in range(N):
        for j in range(i+1, N):
            r = distance(positions[i], positions[j], L)
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

def MoveOneAtom(positions, T, D, L, N):
    origin_positions = positions.copy()
    atom_index = np.random.randint(N)
    displacement = (np.random.rand(2) - 0.5) * D

    positions[atom_index] += displacement

    dE = TotEnergy(positions, L, N) - TotEnergy(origin_positions, L, N)
    if dE < 0 or np.random.rand() < np.exp(- dE / T):
        return positions
    else:
        return origin_positions
    
def MoveAllAtom(positions, T, D, N):
    origin_positions = positions.copy()
    displacement = (np.random.rand(N, 2) - 0.5) * D

    positions += displacement

    dE = TotEnergy(positions, L, N) - TotEnergy(origin_positions, L, N)
    if dE < 0 or np.random.rand() < np.exp(- dE / T):
        return positions
    else:
        return origin_positions
    
# Constants
N = 100  # Number of atoms
T = 0.5 # Temperature
p = 0.96  # Density
L = np.sqrt(N / p)  # Box size
D = 0.3  # Displacement amplitude

# Initialize positions randomly
positions0 = np.random.rand(N, 2) * L

# Plot the atom positions
plt.figure()
plt.scatter(positions0[:, 0], positions0[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Initial Atom Positions')

# Initialize energy
energy = 0.0

positions = positions0.copy()

# Perform Monte Carlo steps
n_steps = 10000
n_accept = 0
E_values = []

with tqdm(total=n_steps) as pbar:
    for step in range(n_steps):
        positions = MoveOneAtom(positions, T, D, L, N)

        # positions = MoveAllAtom(positions, T, D, N)
        
        # Store energy
        E_values.append(TotEnergy(positions, L, N))

        pbar.update()

# Usage
r_values, g_values = calculate_radial_distribution(positions, L)

# Print acceptance ratio
print("Acceptance ratio:", n_accept / n_steps)

# Compute energy and its standard deviation
E_avg = np.mean(E_values)
E_std = np.std(E_values)

print("Average energy:", E_avg)
print("Energy standard deviation:", E_std)

# Plot g(r)
plt.figure()
plt.plot(r_values[:-1], g_values[:-1])
plt.xlabel('r')
plt.ylabel('g(r)')
plt.title('Radial Distribution Function')

# Plot the atom positions
plt.figure()
plt.scatter(positions[:, 0], positions[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Final Atom Positions')

# Show plots
plt.show()