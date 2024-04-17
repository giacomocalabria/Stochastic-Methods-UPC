import numpy as np
import random
import matplotlib.pyplot as plt

def energia_potenziale(posizioni):
  """
  Calcola l'energia potenziale del sistema di particelle.

  Argomenti:
    posizioni: Un array numpy 2D contenente le posizioni delle particelle (x, y).

  Restituisce:
    L'energia potenziale totale del sistema.
  """
  N = len(posizioni)
  energia = 0.0
  for i in range(N):
    for j in range(i + 1, N):
      distanza = np.linalg.norm(posizioni[i] - posizioni[j])
      energia += (posizioni[i][0] * posizioni[j][0]) / distanza
  return energia

def accetta_movimento(delta_energia, temperatura):
  """
  Determina se accettare o meno un movimento proposto.

  Argomenti:
    delta_energia: La variazione di energia associata al movimento proposto.
    temperatura: La temperatura del sistema.

  Restituisce:
    True se il movimento viene accettato, False altrimenti.
  """
  if delta_energia < 0:
    return True
  else:
    probabilita_accettazione = np.exp(-delta_energia / temperatura)
    return random.random() < probabilita_accettazione

def simulazione_monte_carlo(N, temperatura_iniziale, numero_passi, fattore_raffreddamento):
    """
    Esegue una simulazione Monte Carlo per trovare la configurazione ad energia minima.

    Argomenti:
        N: Il numero di particelle nel sistema.
        temperatura_iniziale: La temperatura iniziale del sistema.
        numero_passi: Il numero totale di passi di simulazione.
        fattore_raffreddamento: Il fattore di raffreddamento della temperatura.

    Restituisce:
        La configurazione ad energia minima trovata durante la simulazione.
    """
    # Inizializza le posizioni delle particelle
    posizioni = np.random.rand(N, 2)

    # Inizializza l'energia minima e la configurazione corrispondente
    energia_minima = energia_potenziale(posizioni)
    configurazione_minima = posizioni.copy()

    # Inizializza l'elenco delle energie per il grafico
    energie = [energia_minima]

    temperatura = temperatura_iniziale

    # Esegue i passi di simulazione
    for passo in range(numero_passi):
        # Seleziona una particella casuale
        i = random.randint(0, N - 1)

        # Propone un nuovo movimento per la particella selezionata
        delta_x = random.random() - 0.5
        delta_y = random.random() - 0.5
        nuova_posizione = posizioni[i] + np.array([delta_x, delta_y])

        # Calcola la variazione di energia
        delta_energia = energia_potenziale(nuova_posizione[:, np.newaxis]) - energia_potenziale(posizioni)

        # Accetta o rifiuta il movimento proposto
        if accetta_movimento(delta_energia, temperatura):
            posizioni[i] = nuova_posizione

        # Aggiorna l'energia minima e la configurazione corrispondente se necessario
        if delta_energia < 0:
            energia_minima = energia_potenziale(posizioni)
            configurazione_minima = posizioni.copy()

        # Aggiorna l'elenco delle energie
        energie.append(energia_potenziale(posizioni))

        # Raffredda la temperatura
        temperatura *= fattore_raffreddamento

    # Crea il plot delle particelle prima e dopo la simulazione
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(posizioni[:, 0], posizioni[:, 1], label="Posizioni iniziali")
    plt.title("Posizioni iniziali")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(configurazione_minima[:, 0], configurazione_minima[:, 1], label="Posizioni finali")
    plt.title("Posizioni finali")
    plt.show()

    # Crea il grafico dell'energia potenziale
    plt.figure(figsize=(8, 6))
    plt.plot(range(numero_passi + 1), energie)
    plt.xlabel("Iterazione")
    plt.ylabel("Energia potenziale")
    plt.title("Energia potenziale dopo ogni iterazione")

    # Mostra i plot
    plt.tight_layout()
    plt.show()

    return configurazione_minima

# Esegue un esempio di simulazione
N = 2
temperatura_iniziale = 1.0
numero_passi = 10000
fattore_raffreddamento = 0.99

configurazione_minima = simulazione_monte_carlo(N, temperatura_iniziale, numero_passi, fattore_raffreddamento)

print("Configurazione ad energia minima:")
print(configurazione_minima)

print("Energia minima:")
print(energia_potenziale(configurazione_minima))

