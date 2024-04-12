import numpy as np
import matplotlib.pyplot as plt
import random as rd

# Define the true SNR envelope function (for simulation purposes)
def true_snr_envelope(x):
    return np.sin(2*np.pi*x/100)**2 + 1

# Define the likelihood function (assumed Gaussian)
def likelihood(y, y_obs, sigma_obs):
    return np.exp(-0.5 * ((y - y_obs) / sigma_obs)**2)

# Define the prior distribution for the parameters (here, just standard deviation)
def prior(sigma):
    if sigma < 0:
        return 0
    else:
        return 1

# Define the proposal distribution for MCMC (random walk)
def proposal(sigma_current, sigma_step):
    return np.random.normal(sigma_current, sigma_step)

# Metropolis-Hastings MCMC algorithm
def metropolis_hastings(y_obs, sigma_obs, sigma_initial, sigma_step, n_iterations):
    sigma_chain = [sigma_initial]
    accepted = 0

    for i in range(n_iterations):
        sigma_current = sigma_chain[-1]
        sigma_proposed = proposal(sigma_current, sigma_step)

        # Calculate the acceptance probability
        alpha = likelihood(true_snr_envelope(y_obs), y_obs, sigma_proposed) / \
                likelihood(true_snr_envelope(y_obs), y_obs, sigma_current) * \
                prior(sigma_proposed) / prior(sigma_current)

        # Accept or reject the proposal
        if rd.random() < alpha.any():
            sigma_chain.append(sigma_proposed)
            accepted += 1
        else:
            sigma_chain.append(sigma_current)

    acceptance_rate = accepted / n_iterations
    return sigma_chain, acceptance_rate

# Generate synthetic data
x_values = np.arange(0, 100, 1)
y_obs = true_snr_envelope(x_values) + np.random.normal(0, 0.1, len(x_values))  # Adding noise

# Set initial parameters
sigma_initial = 0.1  # Initial guess for standard deviation
sigma_step = 0.01  # Step size for proposal distribution
n_iterations = 10000  # Number of MCMC iterations

# Run Metropolis-Hastings MCMC
sigma_chain, acceptance_rate = metropolis_hastings(y_obs, 0.1, sigma_initial, sigma_step, n_iterations)

print(f"Acceptance Rate: {acceptance_rate}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sigma_chain)
plt.xlabel('Iteration')
plt.ylabel('Estimated Noise Standard Deviation')
plt.title('Trace Plot of Estimated Noise Standard Deviation')
plt.grid(True)
plt.show()