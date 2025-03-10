import matplotlib.pyplot as plt
import numpy as np

# Dati
strategies = ["FedAvg", "FedProx", "FedNova", "Scaffold", "FedQual", "FedQualAvg"]
values = [3, 0, 1, 1, 4, 7]  # Valori di esempio
colors = ["blue", "green", "red", "orange", "purple", "cyan"]  # Colori per ogni strategia

# Imposta la dimensione della figura
plt.figure(figsize=(10, 6))

# Creazione dell'istogramma
plt.bar(strategies, values, color=colors)

# Aggiunta del titolo e delle etichette degli assi
plt.title("Best Strategies for the scenarios")
plt.xlabel("Strategies")
plt.ylabel("Number of scenarios")
plt.ylim(0, 8)  # Imposta il limite dell'asse y
plt.yticks(np.arange(0, 8, 1))

# Mostra il grafico
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.0)
plt.tight_layout()
plt.show()