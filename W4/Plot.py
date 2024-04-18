import matplotlib.pyplot as plt
import numpy as np

def potential(N, R):
  E = 0
  for i in range(N):
    E += (R[i,0]**2 + R[i,1]**2)  # Termini di energia auto-interazione
    for j in range(i+1, N):
      dij = np.sqrt((R[i,0]-R[j,0])**2 + (R[i,1]-R[j,1])**2)  # Distanza tra particelle i e j
      E += 1 / dij  # Termine di energia di interazione

  return E

# Leggi i dati dai file
risultati_data = np.loadtxt("risultati_20240418-140848.txt")
R = np.loadtxt("configurazione_20240418-140848.txt")
parametri_data = np.loadtxt("parametri_20240418-140848.txt")

# Estrai i dati dall'array parametri_data
N, T0, Tf, dt, steps_per_T, cooling_rate, count, E_min = parametri_data[0:8]
N = int(N)

# Se i dati sono stati salvati come colonne separate, puoi dividerli utilizzando np.hsplit
E = risultati_data[:, 0]
Temp = risultati_data[:, 1]

# Stampa dei parametri
print("Numero di particelle:", N)
print("Temperatura iniziale:", T0)
print("Temperatura finale:", Tf)
print("Passo temporale:", dt)
print("Passi per ogni temperatura:", steps_per_T)
print("Tasso di raffreddamento:", cooling_rate)
print("Numero di passi totali:", count)

# Stampa dell'energia potenziale minima
print("Energia potenziale minima:", min(E))

# Calcolo energia media delle ultime 100 configurazioni
E_avg = np.mean(E[-100:])
print("Energia potenziale media delle ultime 100 configurazioni:", E_avg)

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
plt.grid()
plt.show()

