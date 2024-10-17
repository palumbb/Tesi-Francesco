import pickle
import matplotlib.pyplot as plt

# Lista di etichette per le run
run_labels = ["FedAvg", "FedProx", "FedNova", "Scaffold"]

# Percorso del file .pkl
accuracy_save_path = 'plot_data/consumer/gender_balanced.pkl'

def load_accuracies(file_path):
    """Funzione per caricare i dati di accuracy da un file .pkl."""
    with open(file_path, 'rb') as f:
        accuracies = pickle.load(f)
    return accuracies

def plot_accuracies(accuracies, run_labels):
    """Funzione per plottare le accuracy con le etichette delle run."""
    plt.figure(figsize=(20, 18))
    
    # Per ogni set di accuracy, plotta con l'etichetta corrispondente
    for run_idx, run_accuracies in enumerate(accuracies):
        if run_idx < len(run_labels):
            label = run_labels[run_idx]  # Usa l'etichetta dalla lista
        else:
            label = f"Run {run_idx + 1}"  # Usa una label generica se oltre la lista

        plt.plot(range(1, len(run_accuracies) + 1), run_accuracies, 
                 marker='o', linestyle='-', label=label)

    plt.title('Gender')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Carica i dati dal file .pkl
    accuracies = load_accuracies(accuracy_save_path)

    # Plot dei dati con le etichette delle run
    plot_accuracies(accuracies, run_labels)

if __name__ == "__main__":
    main()
