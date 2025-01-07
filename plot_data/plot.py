import matplotlib.pyplot as plt

#x_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Valori di clients 
x_values = ["Original", "10%", "20%", "30%", "40%", "50%"] # Valori di dirty percentages
#x_values = ["Centralized", 10, 50, 100]

"""y_values = [
    [0.9925, 0.9403, 0.9955, 0.9956, 0.9403, 0.8468, 0.8464, 0.9952, 0.7914, 0.7914, 0.7914],  
    [0.9925, 0.8191, 0.9956, 0.9637, 0.9403, 0.8468, 0.9933, 0.9883, 0.9954, 0.9954, 0.9392], 
    [0.9925, 0.8468, 0.9403, 0.8468, 0.9955, 0.9403, 0.9403, 0.8550, 0.7914, 0.7914, 0.7914], 
    [0.9925, 0.8468, 0.9956, 0.9936, 0.9956, 0.9402, 0.8657, 0.7914, 0.9835, 0.9835, 0.9302]    # Valori di accuracy per ogni strategia
]"""

"""
y_values = [ 
    [0.9679, 0.7596, 0.7507, 0.7835, 0.7461, 0.7110],
    [0.9679, 0.7334, 0.7337, 0.8090, 0.7839, 0.7369]
] 
"""

y_values = [
    [0.8379, 0.6950, 0.5783, 0.6504, 0.5358, 0.6547],  
    [0.7056, 0.7604, 0.5987, 0.6373, 0.6439, 0.7322], 
    [0.7808, 0.7569, 0.6628, 0.6732, 0.5690, 0.6797], 
    [0.6600, 0.6817, 0.6863, 0.6797, 0.6454, 0.6288],
    [0.7381, 0.7959, 0.6875, 0.5578, 0.6550, 0.6905]
]

"""
y_values = [
    [0.9695, 0.6080, 0.9432, 0.8699],
    [0.9698, 0.7083, 0.9467, 0.6724],
    [0.9695, 0.7723, 0.9425, 0.9490],
    [0.9695, 0.7716, 0.9305, 0.6655]
]
"""

labels = ["FedAvg", "FedProx", "FedNova", "Scaffold", "FedQual (β=0.5, γ=0.8)"] 
#labels = ["FedAvg Standard", "FedAvg Mean"]


for y_series, label in zip(y_values, labels):
    plt.plot(x_values, y_series, marker='o', linestyle='-', label=label)

plt.ylabel('Accuracy')
plt.xlabel('Dirty Percentage')
plt.title('Nursery - Unbalanced Standard Imputation')
plt.legend()
plt.grid(True)

# Mostra il grafico
plt.show()