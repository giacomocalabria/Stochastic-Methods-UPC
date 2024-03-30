import random
import matplotlib.pyplot as plt
import numpy as np
import math

Niter = 800
R = 1

# Generate Niter random vectors of length 3 and scale them to the unit cube
y = 2*np.random.rand(Niter, 3)*R-R # Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).

# Count how many of them are inside the unit sphere
Nhint = 0
for i in range(Niter):
    if y[i,0]**2 + y[i,1]**2 + y[i,2]**2 < R**2:
        Nhint += 1

# Estimate the volume of the unit sphere
V = (2*R)**3*Nhint/Niter

print('Estimated volume of the unit sphere:',V)
print('Exact volume of the unit sphere:',4/3*np.pi*R**3)

# Plot the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y[:,0], y[:,1], y[:,2],'.', color='black')

# Plot the sphere
phi = np.linspace(0, np.pi, 100)
theta = np.linspace(0, 2*np.pi, 100)
phi, theta = np.meshgrid(phi, theta)
x = R*np.sin(phi)*np.cos(theta)
y = R*np.sin(phi)*np.sin(theta)
z = R*np.cos(phi)
ax.plot_surface(x, y, z, color='red')
plt.show()