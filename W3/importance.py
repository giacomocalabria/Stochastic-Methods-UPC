import dis
import numpy as np
import math

D = 3  # Numero di dimensioni
num_samples = 1000000  # Numero di campioni
R = 1

samples = np.random.normal(0, R, size=(num_samples, D))

sample_inside = 0
for i in range(num_samples):
    if np.sum(samples[i,:]**2) <= R**2:
        # print("Sample", i, "is inside the hypersphere")
        sample_inside += 1

print("Numero di campioni all'interno dell'ipersfera:", sample_inside)

volume_estimate = (sample_inside / num_samples) * (2 * R)**D
    
print("Stima del volume dell'ipersfera in", D, "dimensioni:", volume_estimate)
print('Exact volume of the unit hypershpere:',np.pi**(D/2)/math.gamma(D/2+1)*R**D)