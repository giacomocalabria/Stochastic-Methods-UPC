import numpy as np
import matplotlib.pyplot as plt

def awgn_channel(signal, snr_dB):
    snr = 10**(snr_dB / 10.0)  # Convert SNR from dB to linear scale
    noise_power = 1.0 / snr
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

def generate_bits(num_bits):
    return np.random.randint(0, 2, num_bits)

def simulate_transmission(num_bits, snr_dB):
    bits_tx = generate_bits(num_bits)
    bits_rx = awgn_channel(bits_tx, snr_dB)
    return bits_tx, bits_rx

def calculate_ber(bits_tx, bits_rx):
    errors = np.sum(bits_tx != np.round(bits_rx))
    ber = errors / len(bits_tx)
    return ber

# Parameters
num_bits = 10000  # Number of bits to transmit
snr_levels_dB = np.linspace(0, 20, 21)  # SNR levels in dB
num_simulations = 100  # Number of simulations per SNR level

# Simulate transmission and calculate BER for each SNR level
ber_envelope = []
for snr_dB in snr_levels_dB:
    total_ber = 0
    for _ in range(num_simulations):
        bits_tx, bits_rx = simulate_transmission(num_bits, snr_dB)
        total_ber += calculate_ber(bits_tx, bits_rx)
    avg_ber = total_ber / num_simulations
    ber_envelope.append(avg_ber)

# Plot the BER envelope
plt.figure(figsize=(10, 6))
plt.semilogy(snr_levels_dB, ber_envelope, marker='o')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER Envelope of AWGN Channel')
plt.grid(True)
plt.show()
