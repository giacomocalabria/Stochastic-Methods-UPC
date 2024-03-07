import random
import matplotlib.pyplot as plt
import numpy as np

Niter = 1000 # Number of random numbers to generate

# Generate N random numbers from 1 to 6
dice_rolls = [random.randint(1, 6) for _ in range(Niter)]

# Count the occurrences of each number
counts = [dice_rolls.count(i) for i in range(1, 7)]

counts = np.array(counts)/Niter

# Plot the results
plt.subplot(1, 2, 1)
plt.bar(range(1, 7), counts)
plt.xlabel(r'Number')
plt.ylabel(r'Count')
plt.title(r'Dice Roll pdf $(N=10^3)$')

Niter = 1000000

# Generate N random numbers from 1 to 6
dice_rolls = [random.randint(1, 6) for _ in range(Niter)]

# Count the occurrences of each number
counts = [dice_rolls.count(i) for i in range(1, 7)]

counts = np.array(counts)/Niter

# Plot the results
plt.subplot(1, 2, 2)
plt.bar(range(1, 7), counts)
plt.xlabel(r'Number')
plt.ylabel(r'Count')
plt.title(r'Dice Roll pdf $(N=10^6)$')

plt.tight_layout()
plt.show()

N = 1000000

# Define the array to store the dice outcomes
dice_outcome = np.zeros(N)

for i in range(N):
    a = random.randint(1, 6)
    b = random.randint(1, 6)
    dice_outcome[i] = (a + b) / 2

# Define the bin edges
bin_edges = np.arange(1, 7, 0.5)

# Count the occurrences of each value
counts, _ = np.histogram(dice_outcome, bins=bin_edges)

# Normalize the counts as a discrete probability density function (PDF)
pdf = counts / N

# Print the counts and PDF
for value, counts, p in zip(bin_edges[:-1], counts, pdf):
    print(f'{value: .1f} {counts: 6d} {p: .5f}')

# Plot the normalized PDF
plt.bar(bin_edges[:-1], pdf, width=0.49)
plt.xlabel(r'Number')
plt.ylabel(r'Probability')
plt.title(r'Dice Roll Distribution (PDF)')
plt.show()