import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider

positions = np.loadtxt("configurazione_20240421-170632.txt")
parametri = np.loadtxt("parametri_20240421-170632.txt")

[N, T, p, D, n_steps, epsi, sigma, TimeComp] = parametri

N = int(N)

L = np.sqrt(N / p)
dr = 0.15
# Print parameters
print("Number of atoms:", N)
print("Temperature:", T)
print("Density:", p)
print("Box size:", L)
print("Number of steps:", n_steps)
print("Epsilon:", epsi)

print("Sigma:", sigma)
print("Time to compute:", TimeComp)

""" # Plot the atom positions
plt.figure()
plt.scatter(positions[:, 0], positions[:, 1])
square = patches.Rectangle((0, 0), L, L, edgecolor='orange', facecolor='none')
plt.gca().add_artist(square)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Final Atom Positions') """

# Function to calculate distance between two points with periodic boundary conditions
distance = lambda r1, r2, L: np.sqrt(((r1[0] - r2[0] - np.round((r1[0] - r2[0]) / L) * L)**2) + ((r1[1] - r2[1] - np.round((r1[1] - r2[1]) / L) * L)**2))

# Compute radial distribution function g(r)
def calculate_radial_distribution(positions, dr, rho, L):
    N = len(positions)
    max_bin_index = int(L / 2 / dr)
    g_values = np.zeros(max_bin_index)

    for i in range(N):
        for j in range(i+1, N):
            r = distance(positions[i], positions[j],L)
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

r_values, g_values = calculate_radial_distribution(positions, dr, rho = p, L = L)

# Initialize the plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
ax.set_xlabel('r')
ax.set_ylabel('g(r)')
ax.set_title('Radial Distribution Function')

# Plot initial g(r)
r_values, g_values = calculate_radial_distribution(positions, dr, rho=p, L=L)
line, = ax.plot(r_values[:-1], g_values[:-1])

# Add Slider
axdr = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
sdr = Slider(axdr, 'dr', 0.005, 0.3, valinit=dr, valstep=0.002)

# Update plot when the slider value changes
def update(val):
    dr_value = sdr.val
    r_values, g_values = calculate_radial_distribution(positions, dr_value, rho=p, L=L)
    line.set_xdata(r_values[:-1])
    line.set_ydata(g_values[:-1])
    ax.set_xlim(min(r_values), max(r_values))
    ax.set_ylim(0, max(g_values))
    fig.canvas.draw_idle()

sdr.on_changed(update)

plt.show()


# Calculate the average energy per atom
sigma = 3.405   # Angstrom (Ar) given in the exercise
epsi = 0.997
lj_potential = lambda r: 4 * epsi * ((sigma / r) ** 12 - (sigma / r) ** 6)

U_values = np.zeros(N*2)
for i in range(N):
    for j in range(i+1, N):
        r = distance(positions[i], positions[j])
        U_values[i] += lj_potential(r)
        U_values[j] += lj_potential(r)

E_avg = np.mean(U_values)
E_std = np.std(U_values)

print("Average energy:", E_avg)
print("Energy standard deviation:", E_std)