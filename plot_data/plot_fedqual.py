import matplotlib.pyplot as plt
import numpy as np

# Dati
strategies = ["FedAvg", "FedProx", "FedNova", "Scaffold", "FedQual"]
accuracy = [0.85, 0.87, 0.88, 0.86, 0.90]  # Valori di esempio
min = np.min(accuracy)
max = np.max(accuracy)
colors = ["blue", "green", "red", "orange", "purple"]  # Colori per ogni strategia

# Conversione delle strategie in numeri per lo stem plot
x = np.arange(len(strategies))

# Creazione dello stem plot con colori diversi
for i, (xi, yi, color) in enumerate(zip(x, accuracy, colors)):
    plt.stem([xi], [yi], basefmt="", linefmt=color, markerfmt=f"{color}")

# Etichette delle x con i nomi delle strategie
plt.xticks(x, strategies)

# Aggiunta del titolo e delle etichette degli assi
plt.title("title")
plt.xlabel("Strategies", labelpad=20)
plt.ylabel("Accuracy")
plt.ylim(min - 0.05, max + 0.05)  # Limita l'intervallo per evidenziare le differenze

# Aggiunta dei valori sopra ogni punto
for i, val in enumerate(accuracy):
    plt.text(x[i], val + 0.005, f"{val:.2f}", ha='center', fontsize=10)

# Mostra il grafico
plt.grid(True, which='both', linestyle='--', linewidth=1.5, alpha=0.25)
plt.tight_layout()
plt.show()
