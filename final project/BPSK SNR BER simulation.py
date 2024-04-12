from cProfile import label
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # for progress bar (optional)
import math
from scipy import special as sp
from scipy.interpolate import interp1d

def qfunc(x):
    return 0.5-0.5*sp.erf(x/math.sqrt(2))

snr_db_start = 5
snr = 10**(snr_db_start / 10)

num_samples = 100000
num_iterations = 140
num_averaging = 20

snr_db_levels = np.zeros([num_iterations, num_averaging])
ber_sim = np.zeros([num_iterations-1, num_averaging])
ber_theor = np.zeros([num_iterations-1, num_averaging])
snr_db_levels[0] = snr_db_start

# Progress bar using tqdm (optional)
with tqdm(total=num_iterations*num_averaging) as pbar:

    for j in range(num_averaging):
        snr_db_levels[0, j] = snr_db_start
        for i in range(1, num_iterations):
            # Progress bar update
            pbar.update()

            transmitted_symbols = np.random.choice([-1, 1], size=num_samples)
            sigma = np.sqrt(1 / (2 * (10**(snr_db_levels[i-1,j] / 10))))
            noise = sigma * np.random.randn(num_samples)
            received_symbols = transmitted_symbols + noise
            num_errors = np.sum(transmitted_symbols != np.sign(received_symbols))
            ber_sim[i-1,j] = num_errors / num_samples
            ber_theor[i-1,j] = qfunc(np.sqrt(2 * (10**(snr_db_levels[i-1,j] / 10))))

            snr_db_proposed = snr_db_levels[i - 1,j] + np.random.normal(scale=0.2)

            likelihood_current = np.exp(-0.5 * np.sum((received_symbols - transmitted_symbols) ** 2) / (10 ** (-snr_db_levels[i-1,j] / 10)))
            likelihood_proposed = np.exp(-0.5 * np.sum((received_symbols - transmitted_symbols) ** 2) / (10 ** (-snr_db_proposed / 10)))

            # Metropolis-Hastings acceptance criterion
            acceptance_prob = min(1, likelihood_proposed / likelihood_current)
            if np.random.rand() < acceptance_prob:
                snr_db_levels[i,j] = snr_db_proposed
            else:
                snr_db_levels[i,j] = snr_db_levels[i - 1,j]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot SNR envelope
ax1.plot(snr_db_levels)
ax1.axhline(y=snr_db_start, color='r', linestyle='--', label="Start SNR")
ax1.set_xlabel("Time (Iterations)")
ax1.set_ylabel("SNR (dB)")
ax1.set_title("SNR Envelope Estimation using MCMC")
ax1.legend()
ax1.grid(True)

# Compute average SNR
snr_db_avg = np.mean(snr_db_levels, axis=1)

# Compute smooth average SNR
f = interp1d(np.arange(num_iterations), snr_db_avg, kind='cubic')
snr_db_smoo = f(np.linspace(0, num_iterations - 1, 1000 * (num_iterations - 1)))

# Plot SNR envelope
#ax2.plot(snr_db_avg, label="Mean SNR (Envelope)")
ax2.plot(np.linspace(0, num_iterations - 1, 1000 * (num_iterations - 1)), snr_db_smoo, label="Smooth Mean SNR (Envelope)")
ax2.axhline(y=snr_db_start, color='r', linestyle='--', label="Start SNR")
ax2.set_xlabel("Time (Iterations)")
ax2.set_ylabel("SNR (dB)")
ax2.set_title("Mean SNR Envelope Estimation using MCMC")
ax2.legend()
ax2.grid(True)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()