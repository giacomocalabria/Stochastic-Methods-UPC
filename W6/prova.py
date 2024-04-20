import numpy as np

# Define constants
hbar = 1.0545718e-34  # Reduced Planck constant (J*s)
mass = 1.0             # Particle mass (kg)
omega = 2*np.pi * 10     # Trap frequency (rad/s)

def harmonic_potential(x):
  """
  Calculates the harmonic potential energy.
  """
  return 0.5 * mass * omega**2 * x**2

def kinetic_energy(p):
  """
  Calculates the kinetic energy.
  """
  return p**2 / (2*mass)

def total_energy(x, p):
  """
  Calculates the total energy.
  """
  return kinetic_energy(p) + harmonic_potential(x)

def metropolis(x, p, beta, step):
  """
  Performs a single Metropolis Monte Carlo step.
  """
  x_new = x + np.random.uniform(-step, step)
  p_new = p + np.random.uniform(-step, step)
  delta_E = total_energy(x_new, p_new) - total_energy(x, p)
  if delta_E <= 0 or np.exp(-beta * delta_E) > np.random.rand():
    return x_new, p_new
  else:
    return x, p

def simulated_annealing(beta_start, beta_end, n_steps, n_equil, n_thermal):
  """
  Performs simulated annealing.
  """
  x = 0.0
  p = 0.0
  beta = beta_start
  positions = []
  for _ in range(n_steps):
    for _ in range(n_equil):
      x, p = metropolis(x, p, beta, 0.1)
    for _ in range(n_thermal):
      x, p = metropolis(x, p, beta, 0.1)
      positions.append(x)
    beta = beta_end + (beta_start - beta_end) * (_ / (n_steps - 1))
  return positions

# Set simulation parameters
beta_start = 0.1
beta_end = 10.0
n_steps = 10000
n_equil = 100
n_thermal = 100

# Run simulation
positions = simulated_annealing(beta_start, beta_end, n_steps, n_equil, n_thermal)

# Calculate density profile
grid_size = 100
grid = np.linspace(-5.0, 5.0, grid_size)
density = np.zeros(grid_size)
for pos in positions:
  density += np.exp(-beta_end * harmonic_potential(grid - pos))

density /= len(positions)

# Print results
print("Ground state energy:", total_energy(positions[0], 0))

# Plot density profile
import matplotlib.pyplot as plt
plt.plot(grid, density)
plt.xlabel("Position")
plt.ylabel("Density")
plt.show()
