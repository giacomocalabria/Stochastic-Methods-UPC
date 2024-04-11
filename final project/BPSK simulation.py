import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # for progress bar (optional)
import math
from scipy import special as sp

def qfunc(x):
    return 0.5-0.5*sp.erf(x/math.sqrt(2))

# Define SNR parameters
snrdb_min = 0
snrdb_max = 10
snrdb = np.arange(snrdb_min, snrdb_max + 0.5, 0.5)  # More precise control over steps
Nsymbols = 10000000  # Number of symbols to simulate

# Define SNR values in dB
snr = 10**(snrdb / 10)

errors = np.zeros_like(snrdb)  # Pre-allocate array for error count

# Progress bar using tqdm (optional)
with tqdm(total=len(snrdb)) as pbar:

    for j, snr_db in enumerate(snrdb):
        # Progress bar update (optional)
        pbar.update()

        sigma = np.sqrt(1 / (2 * snr[j]))  # Noise standard deviation
        error_count = 0

        for _ in range(Nsymbols):  # Simulation loop
            d = np.random.randint(0, 2)  # Generate random data bit (0 or 1)
            x_d = 2 * d - 1  # BPSK modulation: -1 for 0, 1 for 1
            n_d = sigma * np.random.randn()  # Add Gaussian noise
            y_d = x_d + n_d
            if y_d > 0:
                d_est = 1
            else:
                d_est = 0  # Make decision
            if d != d_est:
                error_count += 1  # Count errors

        errors[j] = error_count  # Store error count for each SNR level

ber_sim = errors / Nsymbols  # Simulated BER
ber_theor = qfunc(np.sqrt(2 * snr))  # Theoretical BER (Q-function)

# Plot results
plt.semilogy(snrdb, ber_theor, snrdb, ber_sim, 'o')
plt.axis([snrdb_min, snrdb_max, 1e-6, 1])  # type: ignore # Adjust axis limits for better visualization
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.grid(True)
plt.legend(['Theoretical', 'Simulation'])
plt.title('BPSK Modulation N = 10000000')
plt.show()