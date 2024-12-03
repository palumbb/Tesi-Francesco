import matplotlib.pyplot as plt

x_values = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Valori di clients 

y_values = [
    [0.8242, 0.8388, 0.8324, 0.8351, 0.8360, 0.8379, 0.7527, 0.7554, 0.8287, 0.7527, 0.7554],  
    [0.8223, 0.7536, 0.8406, 0.8379, 0.7573, 0.8369, 0.7545, 0.8287, 0.7564, 0.7554, 0.8333], 
    [0.8241, 0.7554, 0.7545, 0.7545, 0.7564, 0.7538, 0.7545, 0.8342, 0.7527, 0.8324, 0.7527], 
    [0.8214, 0.7811, 0.7545, 0.7666, 0.7545, 0.8379, 0.8232, 0.8245, 0.7564, 0.7426, 0.7500]    # Valori di accuracy per ogni strategia
]

labels = ["FedAvg", "FedProx", "FedNova", "Scaffold"] 

for y_series, label in zip(y_values, labels):
    plt.plot(x_values, y_series, marker='o', linestyle='-', label=label)

plt.xlabel('Accuracy')
plt.ylabel('Number of Dirty Clients')
plt.title('WALL ROBOT NAVIGATION MEAN IMPUTATION 30%')
plt.legend()
plt.grid(True)

# Mostra il grafico
plt.show()