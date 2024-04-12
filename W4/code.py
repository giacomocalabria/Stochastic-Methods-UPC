import random
import math
import numpy as np
import matplotlib.pyplot as plt

def potential(N, R):
    E = 0
    for i in range(N):
        E += R[i,0]**2 + R[i,1]**2
        for j in range(i+1,N):
            E += 1/np.sqrt((R[i,0]-R[j,0])**2 + (R[i,1]-R[j,1])**2)
    return E

def MaxBoltz(N, R, T):
    return math.exp(-potential(N, R)/T)

def localMove(N, R, T, Delta):
    Rp = np.zeros((N,2))

    for j in range(N):
        Rp = R.copy()

        ux = (random.random()*2-1)*Delta
        uy = (random.random()*2-1)*Delta

        Rp[j] = R[j] + [ux,uy]

        # Acceptance probability
        w = MaxBoltz(N, Rp, T)/MaxBoltz(N, R, T)

        # Metropolis test
        if int(random.random() + w) > 0:
            R = Rp
    return R

def globalMode(N, R, T, Delta):
    Rp = np.zeros((N,2))
    
    # Moving all particles
    for i in range(N):
        ux = (random.random()*2-1)*Delta
        uy = (random.random()*2-1)*Delta

        Rp[i] = R[i] + [ux,uy]

    # Acceptance probability
    w = MaxBoltz(N, Rp, T)/MaxBoltz(N, R, T)

    # Metropolis test
    if int(random.random() + w) > 0:
        return R
    else:
        return Rp
    

N = 2 # number of charges
T0 = 10 # Temperatures
Delta = 0.1 # maximum displacement

# Generate a random initial configuration
R0 = np.random.rand(N,2)

R = R0
T = T0

# Number of Monte Carlo steps
nsteps = 10000

Temp = np.zeros(nsteps)
Epot = np.zeros(nsteps)

# Main loop
for i in range(nsteps):

    # Global move
    R = globalMode(N, R, T, Delta)

    # Local move
    R = localMove(N, R, T, Delta)
    
    # Cooling
    T = T * (1 - 1/nsteps)

    # Store temperature and energy
    Temp[i] = T
    Epot[i] = potential(N, R)

# Plot initial and final configurations
plt.plot(R0[:,0],R0[:,1],'o')
plt.plot(R[:,0],R[:,1],'x')
plt.show()

print(R)
print('Final potential energy:', potential(N, R))

# Plot temperature and energy
plt.plot(Temp, label='Temperature')
plt.plot(Epot, label='Potential energy')
plt.legend()
plt.show()