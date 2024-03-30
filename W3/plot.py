import matplotlib.pyplot as plt
import numpy as np
import random
import math

# Make a log-log plot of the statistical error as a function of the number of iterations
N = 100000
R = 1

#I = np.concatenate((np.arange(10, 100, 1), np.arange(100, 300, 3), np.arange(300, 600, 6), np.arange(600, 1000, 10), np.arange(1000, 1200, 12), np.arange(1200, 1500, 15), np.arange(1500, 2000, 20), np.arange(2000, 3000, 30), np.arange(3000, 5000, 50), np.arange(5000, 10000, 100), np.arange(10000, 20000, 200), np.arange(20000, 40000, 400)))
I = np.arange(10, N, 10)
err = np.zeros(len(I)+1)
errS = np.zeros(len(I)+1)

for i in range(len(I)):
    Niter = I[i]
    y = 2*np.random.rand(Niter, 3)*R-R

    Nhint = 0
    for i in range(Niter):
        if y[i,0]**2 + y[i,1]**2 + y[i,2]**2 < R**2:
            Nhint += 1

    V = (2*R)**3*Nhint/Niter
    
    err[i] = np.abs(V-4/3*np.pi*R**3)
    errS[i] = np.sqrt((V-4/3*np.pi*R**3)**2/Niter**2)

# Fit a power law to the data in log-log scale
p = np.polyfit(np.log(I), np.log(errS), 1)

# Plot the power law fit
plt.loglog(I, errS, '.', color='black', label='Data')
plt.loglog(I, np.exp(p[1])*I**p[0], color='red', label='Fit')
plt.xlabel('Number of iterations')
plt.ylabel('Statistical error')
plt.legend()
plt.grid()
plt.show()