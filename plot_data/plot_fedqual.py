import matplotlib.pyplot as plt
import numpy as np

# Dati
strategies = ["FedAvg", "FedProx", "FedNova", "Scaffold", "FedQualAvg", "FedQual"]
accuracy = [0.7869, 0.7837, 0.7950, 0.7900, 0.7836, 0.8032]  # Valori di esempio
min = np.min(accuracy)
max = np.max(accuracy)
colors = ["blue", "green", "red", "orange", "purple", "black"]  # Colori per ogni strategia

# Conversione delle strategie in numeri per lo stem plot
x = np.arange(len(strategies))

# Creazione dello stem plot con colori diversi
for i, (xi, yi, color) in enumerate(zip(x, accuracy, colors)):
    plt.stem([xi], [yi], basefmt="", linefmt=color, markerfmt=f"{color}")

# Etichette delle x con i nomi delle strategie
plt.xticks(x, strategies)

# Aggiunta del titolo e delle etichette degli assi
plt.title("HEART - Small Clean VS Big Dirty Clients")
plt.xlabel("Strategies", labelpad=20)
plt.ylabel("Accuracy")
plt.ylim(min - 0.02, max + 0.02)  # Limita l'intervallo per evidenziare le differenze

# Aggiunta dei valori sopra ogni punto
for i, val in enumerate(accuracy):
    plt.text(x[i], val + 0.002, f"{val}", ha='center', fontsize=8.5)

# Mostra il grafico
plt.grid(True, which='both', linestyle='--', linewidth=1.5, alpha=0.25)
plt.tight_layout()
plt.show()
