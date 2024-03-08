import numpy as np

def f_exp(x):
    return np.exp(-x)

def f_condition(x):
    if x**2 > 0.25:
        return x
    else:
        return 0.5

def monte_carlo(n_samples):
    sum_exp = 0.0
    sum_condition = 0.0

    for _ in range(n_samples):
        x = np.random.rand()
        sum_exp += f_exp(x)
        sum_condition += f_condition(x)

    expectation_exp = sum_exp / n_samples
    expectation_condition = sum_condition / n_samples

    return expectation_exp, expectation_condition

def main():
    n_samples = 1000000
    expectation_exp, expectation_condition = monte_carlo(n_samples)

    with open('monte_carlo_results.txt', 'w') as file:
        file.write(f'Expectation value of Exp(-x): {expectation_exp}\n')
        file.write(f'Expectation value of x if x^2 > 1/4, 1/2 otherwise: {expectation_condition}\n')

if __name__ == "__main__":
    main()

