import matplotlib.pyplot as plt

x_values = [10, 50, 100]  # Valori di clients 

y_values = [
    [10, 15, 13],  
    [12, 17, 14], 
    [3, 10, 25], 
    [8, 13, 12]    # Valori di accuracy per ogni strategia
]

labels = ["FedAvg", "FedProx", "FedNova", "Scaffold"] 

for y_series, label in zip(y_values, labels):
    plt.plot(x_values, y_series, marker='o', linestyle='-', label=label)

plt.xlabel('Clients')
plt.ylabel('Accuracy')
plt.title('Grafico con sottoliste e legende')
plt.legend()
plt.grid(True)

# Mostra il grafico
plt.show()