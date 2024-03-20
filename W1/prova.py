import numpy as np
import matplotlib.pyplot as plt

# Funzione per calcolare la media di 1000 lanci di un dado
def media_dado():
    return np.mean(np.random.randint(1, 7, size=1000))

# Genera 10000 medie di 1000 lanci di un dado
medie = [media_dado() for _ in range(100000)]

# Calcola la distribuzione di probabilità
hist, bins = np.histogram(medie, bins=50, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Normalizza la distribuzione
area = np.sum(hist * np.diff(bins))
hist /= area

# Plot della distribuzione normalizzata
plt.plot(bin_centers, hist, label='Probability Distribution')
plt.xlabel('Media dei lanci del dado')
plt.ylabel('Densità di probabilità')
plt.title('Distribuzione di probabilità della media di 1000 lanci di un dado')
plt.legend()
plt.grid(True)
plt.show()