import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_samples = 100000  # Number of transmitted symbols
snr_db_true = 5  # True SNR in dB (constant over time)
num_iterations = 100  # Number of MCMC iterations

# Generate binary data (e.g., BPSK symbols)
transmitted_symbols = np.random.choice([-1, 1], size=num_samples)

# AWGN channel simulation (add noise)
noise_power = 10 ** (-snr_db_true / 10)  # Convert dB to linear scale
noise = np.sqrt(noise_power) * np.random.randn(num_samples)

# Received signal after passing through AWGN channel
received_signal = transmitted_symbols + noise

# Initialize SNR envelope (MCMC parameter)
snr_db_estimate = np.zeros(num_iterations)
snr_db_estimate[0] = snr_db_true  # Initial guess (can be improved)

# MCMC simulation
for i in range(1, num_iterations):
    # Propose a new SNR value (random walk proposal)
    snr_db_proposed = snr_db_estimate[i - 1] + np.random.normal(scale=0.2)

    # Calculate likelihood (assuming Gaussian noise)
    likelihood_current = np.exp(-0.5 * np.sum((received_signal - transmitted_symbols) ** 2) / noise_power)
    likelihood_proposed = np.exp(-0.5 * np.sum((received_signal - transmitted_symbols) ** 2) / (10 ** (-snr_db_proposed / 10)))

    # Metropolis-Hastings acceptance criterion
    acceptance_prob = min(1, likelihood_proposed / likelihood_current)
    if np.random.rand() < acceptance_prob:
        snr_db_estimate[i] = snr_db_proposed
    else:
        snr_db_estimate[i] = snr_db_estimate[i - 1]

# Plot SNR envelope
plt.figure(figsize=(8, 4))
plt.plot(snr_db_estimate, label="Estimated SNR (Envelope)")
plt.axhline(y=snr_db_true, color='r', linestyle='--', label="True SNR")
plt.xlabel("Time (Iterations)")
plt.ylabel("SNR (dB)")
plt.title("SNR Envelope Estimation using MCMC")
plt.legend()
plt.grid(True)
plt.show()