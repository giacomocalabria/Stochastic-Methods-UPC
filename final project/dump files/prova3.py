import numpy as np
import matplotlib.pyplot as plt

def awgn_channel(signal, snr_dB):
    """
    Simulates an AWGN channel by adding noise to the signal.

    Parameters:
        signal (array_like): Input signal.
        snr_dB (float): Signal-to-Noise Ratio in dB.

    Returns:
        array_like: Noisy signal.
    """
    snr = 10**(snr_dB / 10.0)  # Convert SNR from dB to linear scale
    noise_power = 1.0 / snr
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

def generate_bits(num_bits):
    """
    Generates random binary bits.

    Parameters:
        num_bits (int): Number of bits to generate.

    Returns:
        array_like: Array of randomly generated bits.
    """
    return np.random.randint(0, 2, num_bits)

def likelihood(bits_tx, bits_rx):
    """
    Calculates the likelihood of observing received bits given transmitted bits.

    Parameters:
        bits_tx (array_like): Transmitted bits.
        bits_rx (array_like): Received bits.

    Returns:
        float: Likelihood.
    """
    return np.prod(bits_tx == np.round(bits_rx))

def prior(snr_current, snr_proposed, snr_step):
    """
    Calculates the prior probability for SNR.

    Parameters:
        snr_current (float): Current SNR.
        snr_proposed (float): Proposed SNR.
        snr_step (float): Step size for the proposal distribution.

    Returns:
        float: Prior probability.
    """
    return np.exp(-0.5 * ((snr_proposed - snr_current) / snr_step)**2)

def proposal(snr_current, snr_step):
    """
    Generates a proposed SNR value.

    Parameters:
        snr_current (float): Current SNR.
        snr_step (float): Step size for the proposal distribution.

    Returns:
        float: Proposed SNR.
    """
    return snr_current + snr_step * np.random.randn()

def metropolis_hastings_mcmc(num_bits, snr_levels_dB, num_iterations, snr_step):
    """
    Metropolis-Hastings MCMC algorithm to estimate the BER envelope.

    Parameters:
        num_bits (int): Number of bits to transmit.
        snr_levels_dB (array_like): Array of SNR levels in dB.
        num_iterations (int): Number of MCMC iterations.
        snr_step (float): Step size for SNR proposal distribution.

    Returns:
        array_like: Array of BER values over time for each SNR level.
    """
    ber_envelope = []
    for snr_dB in snr_levels_dB:
        snr_chain = [snr_dB]
        accepted = 0
        for _ in range(num_iterations):
            snr_current = snr_chain[-1]
            snr_proposed = proposal(snr_current, snr_step)
            bits_tx = generate_bits(num_bits)
            bits_rx = awgn_channel(bits_tx, snr_proposed)
            alpha = likelihood(bits_tx, bits_rx) * \
                    prior(snr_current, snr_proposed, snr_step) / \
                    (likelihood(bits_tx, bits_rx) * \
                     prior(snr_proposed, snr_current, snr_step))
            if np.random.rand() < alpha:
                snr_chain.append(snr_proposed)
                accepted += 1
            else:
                snr_chain.append(snr_current)
        ber_envelope.append(accepted / num_iterations)
    return ber_envelope

# Parameters
num_bits = 10000  # Number of bits to transmit
snr_levels_dB = np.linspace(0, 20, 21)  # SNR levels in dB
num_iterations = 10000  # Number of MCMC iterations per SNR level
snr_step = 0.1  # Step size for SNR proposal distribution

# Run Metropolis-Hastings MCMC to compute the envelope over time of the BER
ber_envelope = metropolis_hastings_mcmc(num_bits, snr_levels_dB, num_iterations, snr_step)

# Plot the BER envelope over SNR levels
plt.figure(figsize=(10, 6))
plt.plot(snr_levels_dB, ber_envelope, marker='o')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER Envelope of AWGN Channel over SNR Levels')
plt.grid(True)
plt.show()
