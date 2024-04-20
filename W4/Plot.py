import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

def potential(N, R):
  E = 0
  for i in range(N):
    E += (R[i,0]**2 + R[i,1]**2)  # Termini di energia auto-interazione
    for j in range(i+1, N):
      dij = np.sqrt((R[i,0]-R[j,0])**2 + (R[i,1]-R[j,1])**2)  # Distanza tra particelle i e j
      E += 1 / dij  # Termine di energia di interazione

  return E

# Leggi i dati dai file
risultati_data = np.loadtxt("risultati_20240418-174346.txt")
R = np.loadtxt("configurazione_20240418-174346.txt")
parametri_data = np.loadtxt("parametri_20240418-174346.txt")

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

# Trova le coordinate della particella più vicina all'origine
R_closest = R[np.argmin(R[:,0]**2 + R[:,1]**2)]
print("Coordinate della particella più vicina all'origine:", R_closest)

# Calcola la distanza media dall'origine
R_avg = np.mean(np.sqrt(R[:,0]**2 + R[:,1]**2))
print("Distanza media dall'origine:", R_avg)

#Calcola la distanza media dall' origine delle particelle che sono oltre il raggio R_avg
R_avg_out = np.mean(np.sqrt(R[np.sqrt(R[:,0]**2 + R[:,1]**2) > R_avg,0]**2 + R[np.sqrt(R[:,0]**2 + R[:,1]**2) > R_avg,1]**2))
print("Distanza media dall'origine delle particelle che sono oltre il raggio R_avg:", R_avg_out)

#Calcola la distanza media dall' origine delle particelle che sono entro il raggio R_avg ma maggiori di R_avg/2
#perchè altrimenti sarebbero le stesse delle precedenti perchè la più vicina all'origine è anche la più
#vicina al raggio medio e la più vicina al raggio medio è anche la più vicina all'origine
#quindi prendo le particelle che sono entro il raggio R_avg ma che non sono le più vicine all'origine

# Seleziono le particelle che sono entro il raggio R_avg
Rin = R[np.sqrt(R[:,0]**2 + R[:,1]**2) > R_avg/2]
Rin = Rin[np.sqrt(Rin[:,0]**2 + Rin[:,1]**2) < R_avg]

# Calcolo la distanza media dall'origine delle particelle in Rin
R_avg_in = np.mean(np.sqrt(Rin[:,0]**2 + Rin[:,1]**2))
print("Distanza media dall'origine delle particelle che sono entro il raggio R_avg:", R_avg_in)

# Calcola la distanza media dall'origine del primo livello sotto R_avg_in/2
Rinner = R[np.sqrt(R[:,0]**2 + R[:,1]**2) < R_avg_in/2]

# Calcolo la distanza media dall'origine delle particelle in Rinner
R_avg_inner = np.mean(np.sqrt(Rinner[:,0]**2 + Rinner[:,1]**2))
print("Distanza media dall'origine delle particelle che sono entro il raggio R_avg_in/2:", R_avg_inner)

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
ax2.set_ylabel('Temperature', color=color)
ax2.plot(Temp, color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.show()

# Visualizzazione della configurazione ottimale e di un cerchio di raggio R_avg
plt.figure()
plt.scatter(R[:,0], R[:,1])
circle1 = Circle((0, 0), R_avg, color='r', fill=False)
circle2 = Circle((0, 0), R_avg_out, color='g', fill=False)
circle3 = Circle((0, 0), R_avg_in, color='r', fill=False)
circle4 = Circle((0, 0), R_avg_inner, color='orange', fill=False)
plt.gca().add_artist(circle2)
plt.gca().add_artist(circle3)
plt.gca().add_artist(circle4)
plt.axis('equal')
plt.title("Optimal configuration of N = " + str(N) + " particles")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

''''plt.figure()
plt.scatter(R[:,0], R[:,1])
plt.title("Optimal configuration of N = " + str(N) + " particles")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()'''

