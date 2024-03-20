import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

N = 100000  #number of random events
W = 1000  #number of dice rolls per event

# Funzione per calcolare la media di 1000 lanci di un dado
def media_dado():
    return np.mean(np.random.randint(1, 7, size=W))

# Genera 10000 medie di 1000 lanci di un dado
medie = [media_dado() for _ in range(N)]

# Calcola la distribuzione di probabilità
hist, bins = np.histogram(medie, bins=50, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Plot della distribuzione normale
plt.plot(bin_centers, hist, label='Distribuzione calcolata')

# Plot della distribuzione gaussiana con media 3.5
x = np.linspace(3, 4, 1000)
gaussian = norm.pdf(x, loc=3.5, scale=np.std(medie))
plt.plot(x, gaussian, label='Distribuzione gaussiana',color='red', linestyle='--')

plt.xlabel('Media dei lanci del dado')
plt.ylabel('Densità di probabilità')
plt.title('Distribuzione di probabilità della media di 1000 lanci di un dado')
plt.legend()
plt.grid(True)
plt.show()