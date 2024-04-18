import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm  # for progress bar (optional)
import time
import os

def potential(N, R):
  E = 0
  for i in range(N):
    E += (R[i,0]**2 + R[i,1]**2)  # Termini di energia auto-interazione
    for j in range(i+1, N):
      dij = np.sqrt((R[i,0]-R[j,0])**2 + (R[i,1]-R[j,1])**2)  # Distanza tra particelle i e j
      E += 1 / dij  # Termine di energia di interazione

  return E

def MaxBoltz(dE, T):
    return math.exp(-(dE)/T)

def MoveOneParticle(N,R,T,dt):
    # Random choice of a charge to move
    i = np.random.randint(N)

    # Do a random displacement
    R_new = R.copy()
    R_new[i] += np.random.rand(2) * dt - dt/2

    # Potential energy difference calculation
    dE = potential(N, R_new) - potential(N, R)

    # Acceptance or rejection of the new position based on the metropolis algorithm
    if dE < 0:
      return R_new
    else:
      p_acc = MaxBoltz(dE, T)
      if np.random.rand() < p_acc:
        return R_new
      else:
        return R
      
def MoveAllParticles(N,R,T,dt):
    R_new = R.copy()
    for i in range(N):
      R_new[i] += np.random.rand(2) * dt - dt/2
    dE = potential(N, R_new) - potential(N, R)
    if dE < 0:
      return R_new
    else:
      p_acc = MaxBoltz(dE, T)
      if np.random.rand() < p_acc:
        return R_new
      else:
        return R

# Parametri di simulazione
N = 26   # Numero di particelle
T0 = 10  # Temperatura iniziale
Tf = 0.01  # Temperatura finale
dt = 0.5  # Passo temporale
steps_per_T = 1000  # Numero di passi per ogni temperatura
cooling_rate = 0.995  # Tasso di raffreddamento
count = math.ceil(math.log(Tf / T0) / math.log(cooling_rate))

# Inizializzazione casuale delle posizioni
R0 = np.random.rand(N, 2) * 2 - 1

Temp = np.zeros((count))
E = np.zeros((count))

T = T0
R = R0.copy()
i = 0

# Visualizzazione parametri
print("Numero di particelle:", N)
print("Temperatura iniziale:", T0)
print("Temperatura finale:", Tf)
print("Passo temporale:", dt)
print("Passi per ogni temperatura:", steps_per_T)
print("Tasso di raffreddamento:", cooling_rate)
print("Numero di passi totali:", count)

Tstart = time.time()
with tqdm(total=count) as pbar:
    # Simulazione del raffreddamento 
    
    while T > Tf:
      pbar.update()
      for _ in range(steps_per_T):
        R = MoveOneParticle(N, R, T, dt)

      for _ in range(100):
        R = MoveAllParticles(N, R, T, dt)

      T *= cooling_rate

      Temp[i] = T
      E[i] = potential(N, R)
      i += 1

Tend = time.time()
print("Tempo di esecuzione:", Tend - Tstart)


####################################
## ANALISI E STAMPA DEI RISULTATI ##
####################################

# Stampa dell'energia potenziale minima
E_min = potential(N, R)
print("Energia potenziale minima:", E_min)

# Creazione della cartella se non esiste già
output_folder = "Outputs"
os.makedirs(output_folder, exist_ok=True)

# Salvataggio dei dati su file all'interno della cartella
np.savetxt(os.path.join(output_folder, "risultati_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"), np.column_stack((E, Temp)))
np.savetxt(os.path.join(output_folder, "configurazione_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"), R)
np.savetxt(os.path.join(output_folder, "parametri_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"), [int(N), T0, Tf, dt, steps_per_T, cooling_rate, count, E_min])

# Creazione della figura
fig, ax1 = plt.subplots()

# Plot dell'energia potenziale sull'asse sinistro
color = 'tab:red'
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Potential Energy', color=color)
ax1.plot(E, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Creazione dell'asse per la temperatura sull'asse destro
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Temperature', color=color)
ax2.plot(Temp, color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.show()

# Visualizzazione della configurazione ottimale
plt.figure()
plt.scatter(R[:,0], R[:,1])
plt.title("Optimal configuration of N = " + str(N) + " particles")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
