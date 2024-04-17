import numpy as np
import matplotlib.pyplot as plt
import math

def potential(N, R):
  E = 0
  for i in range(N):
    E += (R[i,0]**2 + R[i,1]**2)  # Termini di energia auto-interazione
    for j in range(i+1, N):
      dij = np.sqrt((R[i,0]-R[j,0])**2 + (R[i,1]-R[j,1])**2)  # Distanza tra particelle i e j
      E += 1 / dij  # Termine di energia di interazione

  return E

def MaxBoltz(N, R, Rp, T):
    return math.exp(-(potential(N,Rp) - potential(N, R))/T)

def simulate_annealing(N, R0, T0, Tf, dt, steps_per_T, cooling_rate):
  R = R0.copy()
  T = T0

  Temp = np.array([])
  E = np.array([])

  # Simulazione del raffreddamento 
  while T > Tf:
    for _ in range(steps_per_T):
      # Scelta casuale di una particella da spostare
      i = np.random.randint(N)

      # Nuova posizione proposta
      R_new = R.copy()
      R_new[i] += np.random.rand(2) * dt - dt/2

      # Calcolo della differenza di energia potenziale
      dE = potential(N, R_new) - potential(N, R)

      # Accettazione o rifiuto della nuova posizione
      if dE < 0:
        R = R_new
      else:
        p_acc = MaxBoltz(N, R, R_new, T)
        if np.random.rand() < p_acc:
          R = R_new

    # Raffreddamento
    T *= cooling_rate

    # Salvataggio della temperatura e dell'energia
    Temp = np.append(Temp, T)
    E = np.append(E, potential(N, R))
  
  # Visualizzazione dell'energia potenziale dopo ogni passo
  plt.plot(E)
  plt.plot(Temp)
  plt.title("Energia potenziale durante la simulazione")
  plt.show()

  return R

if __name__ == "__main__":
  # Parametri di simulazione
  N = 20  # Numero di particelle
  T0 = 10  # Temperatura iniziale
  Tf = 0.01  # Temperatura finale
  dt = 0.5  # Passo temporale
  steps_per_T = 1000  # Numero di passi per ogni temperatura
  cooling_rate = 0.99  # Tasso di raffreddamento

  # Inizializzazione casuale delle posizioni
  R0 = 2 * np.random.rand(N, 2) - 1

  # Esecuzione della simulazione
  R_opt = simulate_annealing(N, R0, T0, Tf, dt, steps_per_T, cooling_rate)

  # Stampa dell'energia potenziale minima
  E_min = potential(N, R_opt)
  print("Energia potenziale minima:", E_min)

  # Visualizzazione della configurazione ottimale
  plt.scatter(R_opt[:,0], R_opt[:,1])
  plt.title("Configurazione ottimale di N particelle")
  plt.show()

