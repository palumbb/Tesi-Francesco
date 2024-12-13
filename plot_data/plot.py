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

y_values = [ 
    [0.9679, 0.7596, 0.7507, 0.7835, 0.7461, 0.7110],
    [0.9679, 0.7334, 0.7337, 0.8090, 0.7839, 0.7369]
]

"""y_values = [
    [0.9170, 0.8463, 0.8024, 0.7990],  
    [0.9170, 0.8780, 0.7926, 0.7902], 
    [0.9170, 0.8850, 0.7926, 0.7804], 
    [0.9170, 0.8524, 0.7877, 0.7170] 
]"""

#labels = ["FedAvg", "FedProx", "FedNova", "Scaffold"] 
labels = ["FedAvg Standard", "FedAvg Mean"]


for y_series, label in zip(y_values, labels):
    plt.plot(x_values, y_series, marker='o', linestyle='-', label=label)

plt.ylabel('Accuracy')
plt.xlabel('Dirty Percentage')
plt.title('Nursery - Standard Proportions = [2%, 8%, 5%, 5%, 10%, 30%, 15%, 5%, 10%, 10%]')
plt.legend()
plt.grid(True)

# Mostra il grafico
plt.show()