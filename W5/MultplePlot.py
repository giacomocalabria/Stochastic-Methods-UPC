import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
import scipy.constants

positions05 = np.loadtxt("configurazione_20240421-170632.txt")
parametri05 = np.loadtxt("parametri_20240421-170632.txt")

positions10 = np.loadtxt("configurazione_20240421-170654.txt")
parametri10 = np.loadtxt("parametri_20240421-170654.txt")

positions15 = np.loadtxt("configurazione_20240421-170724.txt") # OK
parametri15 = np.loadtxt("parametri_20240421-170724.txt")

positions20 = np.loadtxt("configurazione_20240421-170704.txt")
parametri20 = np.loadtxt("parametri_20240421-170704.txt")

positions25 = np.loadtxt("configurazione_20240421-170742.txt") # OK
parametri25 = np.loadtxt("parametri_20240421-170742.txt")

positions30 = np.loadtxt("configurazione_20240421-170712.txt") # OK
parametri30 = np.loadtxt("parametri_20240421-170713.txt")

N = int(parametri05[0])
p = parametri05[2]
L = np.sqrt(N / p)
dr = 0.1
epsi = parametri05[5]
sigma = parametri05[6]

T = [parametri05[1], parametri10[1], parametri15[1], parametri20[1], parametri25[1], parametri30[1]]

# Function to calculate distance between two points with periodic boundary conditions
def distance(r1, r2):
    dx = r1[0] - r2[0]
    dy = r1[1] - r2[1]
    dx -= np.round(dx / L) * L
    dy -= np.round(dy / L) * L
    return np.sqrt(dx**2 + dy**2)

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

r_values = np.zeros([6, int(L / 2 / dr)])
g_values = np.zeros([6, int(L / 2 / dr)])

r_values[0], g_values[0] = calculate_radial_distribution(positions05, dr, p, L)
r_values[1], g_values[1] = calculate_radial_distribution(positions10, dr, p, L)
r_values[2], g_values[2] = calculate_radial_distribution(positions15, dr, p, L)
r_values[3], g_values[3] = calculate_radial_distribution(positions20, dr, p, L)
r_values[4], g_values[4] = calculate_radial_distribution(positions25, dr, p, L)
r_values[5], g_values[5] = calculate_radial_distribution(positions30, dr, p, L)

""" plt.figure()
plt.plot(r_values[0], g_values[0])
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Radial distribution function g(r) for T = 0.5")

plt.figure()
plt.plot(r_values[1], g_values[1])
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Radial distribution function g(r) for T = 1.0")

plt.figure()
plt.plot(r_values[2], g_values[2])
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Radial distribution function g(r) for T = 1.5")

plt.figure()
plt.plot(r_values[3], g_values[3])
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Radial distribution function g(r) for T = 2.0")

plt.figure()
plt.plot(r_values[4], g_values[4])
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Radial distribution function g(r) for T = 2.5")

plt.figure()
plt.plot(r_values[5], g_values[5])
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Radial distribution function g(r) for T = 3.0")

plt.show() """

fig, ax = plt.subplots()
ax.plot(r_values[0], g_values[0], label=f"T = {T[0]}")
#ax.plot(r_values[1], g_values[1], label=f"T = {T[1]}")
#ax.plot(r_values[2], g_values[2], label=f"T = {T[2]}")
#ax.plot(r_values[3], g_values[3], label=f"T = {T[3]}")
#ax.plot(r_values[4], g_values[4], label=f"T = {T[4]}")
ax.plot(r_values[5], g_values[5], label=f"T = {T[5]}")
ax.set_xlabel("r")
ax.set_ylabel("g(r)")
ax.legend()
plt.show()

# Calculate the average energy per atom at different temperatures
sigma = 3.405   # Angstrom (Ar) given in the exercise
epsi = 0.997
lj_potential = lambda r: 4 * epsi * ((sigma / r) ** 12 - (sigma / r) ** 6)

U_values = np.zeros(N*2)
for i in range(N):
    for j in range(i+1, N):
        r = distance(positions05[i], positions05[j])
        U_values[i] += lj_potential(r)
        U_values[j] += lj_potential(r)

E_avg = np.mean(U_values)
E_std = np.std(U_values)

print("Average energy 0.5:", E_avg)
print("Energy standard deviation 0.5:%.2f", E_std)

U_values = np.zeros(N*2)
for i in range(N):
    for j in range(i+1, N):
        r = distance(positions10[i], positions10[j])
        U_values[i] += lj_potential(r)
        U_values[j] += lj_potential(r)

E_avg = np.mean(U_values)
E_std = np.std(U_values)

print("Average energy 1:%.2f", E_avg)
print("Energy standard deviation 1:%.2f", E_std)

U_values = np.zeros(N*2)
for i in range(N):
    for j in range(i+1, N):
        r = distance(positions15[i], positions15[j])
        U_values[i] += lj_potential(r)
        U_values[j] += lj_potential(r)

E_avg = np.mean(U_values)
E_std = np.std(U_values)

print("Average energy 1.5:", E_avg)
print("Energy standard deviation 1.5:", E_std)

U_values = np.zeros(N*2)
for i in range(N):
    for j in range(i+1, N):
        r = distance(positions20[i], positions20[j])
        U_values[i] += lj_potential(r)
        U_values[j] += lj_potential(r)

E_avg = np.mean(U_values)
E_std = np.std(U_values)

print("Average energy 2:", E_avg)
print("Energy standard deviation 2:", E_std)

U_values = np.zeros(N*2)
for i in range(N):
    for j in range(i+1, N):
        r = distance(positions25[i], positions25[j])
        U_values[i] += lj_potential(r)
        U_values[j] += lj_potential(r)

E_avg = np.mean(U_values)
E_std = np.std(U_values)

print("Average energy 2.5:", E_avg)
print("Energy standard deviation 2.5:", E_std)

U_values = np.zeros(N*2)
for i in range(N):
    for j in range(i+1, N):
        r = distance(positions30[i], positions30[j])
        U_values[i] += lj_potential(r)
        U_values[j] += lj_potential(r)

E_avg = np.mean(U_values)
E_std = np.std(U_values)

print("Average energy 3:", E_avg)
print("Energy standard deviation 3:", E_std)

from scipy.signal import savgol_filter
yhat = savgol_filter(g_values[0], 6, 4) # window size 51, polynomial order 3
plt.figure()
plt.plot(r_values[0], yhat, color='red')
#plt.plot(r_values[0], g_values[0], color='blue')
plt.xlabel("r")
plt.ylabel("g(r)")
plt.show()


