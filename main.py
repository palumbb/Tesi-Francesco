"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import os
import pickle
import matplotlib.pyplot as plt
import flwr as fl
import hydra
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
from model.binarynet import BinaryNet, train_centralized_binary, test_binary
from model.multiclassnet import MulticlassNet, train_centralized_multi, test_multi
from data import load_dataset
from servers.server_fednova import FedNovaServer
from servers.server_scaffold import ScaffoldServer, gen_evaluate_fn
from strategy import FedNovaStrategy, ScaffoldStrategy
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
import pandas as pd


@hydra.main(config_path="conf", config_name="fedavg_base", version_base=None)

def main(cfg: DictConfig) -> None:


    device = cfg.server_device

    print(f"Strategy: {cfg.name}")
    print("Dataset: " + str(cfg.dataset_path))
    print("Partitioning: " + str(cfg.partitioning))
    print("Clients:" + str(cfg.num_clients))
    print("Local epochs:" + str(cfg.num_epochs))
    print("Sampled clients: " + str(cfg.clients_per_round))
    print("Rounds: " + str(cfg.num_rounds))

    accuracies = []
    run_labels = ["FedAvg", "FedProx", "FedNova", "Scaffold"]

    """if cfg.dataset_path=="./data/consumer.csv":
        accuracy_save_path = 'plot_data/consumer/10clients.pkl'
    elif cfg.dataset_path=="./data/mv.csv":
        accuracy_save_path = 'plot_data/mv/10clients.pkl'
    elif cfg.dataset_path=="./data/shuttle.csv":
        accuracy_save_path = 'plot_data/shuttle/10clients.pkl'
    elif cfg.dataset_path=="./data/nursery.csv":
        accuracy_save_path = 'plot_data/nursery/10clients.pkl'
    elif cfg.dataset_path=="./data/mushrooms.csv":
        accuracy_save_path = 'plot_data/mushrooms/10clients.pkl'
    elif cfg.dataset_path=="./data/wall-robot-navigation.csv":
        accuracy_save_path = 'plot_data/wall-robot-navigation/10clients.pkl'
    elif cfg.dataset_path=="./data/car.csv":
        accuracy_save_path = 'plot_data/car/10clients.pkl'

    current_run_idx = len(load_accuracies(accuracy_save_path))  
    if current_run_idx < len(run_labels):
        run_label = run_labels[current_run_idx]
    else:
        run_label = f"Run {current_run_idx + 1}" 
    """

    # 2. Prepare your dataset
    if cfg.federated:
        trainloaders, valloaders, testloader = load_dataset(
            data_cfg=cfg.dataset,
            num_clients=cfg.num_clients,
            federated=cfg.federated,
            partitioning=cfg.partitioning,
            model=cfg.model
        )

        # 3. Define your clients
        client_fn = None
        if cfg.client_fn._target_ == "clients.multiclass.client_scaffold.gen_client_fn":
            save_path = HydraConfig.get().runtime.output_dir
            client_cv_dir = os.path.join(save_path, "client_cvs")
            print("Local cvs for scaffold clients are saved to: ", client_cv_dir)
            client_fn = call(
                cfg.client_fn,
                trainloaders,
                valloaders,
                model=cfg.model,
                client_cv_dir=client_cv_dir,
            )
        else:
            client_fn = call(
                cfg.client_fn,
                trainloaders,
                valloaders,
                model=cfg.model,
            )

        evaluate_fn = gen_evaluate_fn(
            testloader, 
            device=device, 
            model=cfg.model, 
            accuracies=accuracies,
        )

        # 4. Define your strategy
        strategy = instantiate(
            cfg.strategy,
            evaluate_fn=evaluate_fn,
        )

        # 5. Define your server
        server = Server(strategy=strategy, client_manager=SimpleClientManager())
        if isinstance(strategy, FedNovaStrategy):
            server = FedNovaServer(strategy=strategy, client_manager=SimpleClientManager())
        elif isinstance(strategy, ScaffoldStrategy):
            server = ScaffoldServer(
                strategy=strategy, model=cfg.model, client_manager=SimpleClientManager()
            )

        # 6. Start Simulation
        history = fl.simulation.start_simulation(
            server=server,
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            client_resources={
                "num_cpus": cfg.client_resources.num_cpus,
                "num_gpus": cfg.client_resources.num_gpus,
            },
            strategy=strategy,
        )

        print(history)
        save_path = HydraConfig.get().runtime.output_dir
        print(save_path)

        # 7. Save your results
        with open(os.path.join(save_path, "history.pkl"), "wb") as f_ptr:
            pickle.dump(history, f_ptr)

        #save_accuracies(accuracies, accuracy_save_path)

        # 8. Plot the accuracies for each strategy
        """all_accuracies = load_accuracies(accuracy_save_path)

        plt.figure(figsize=(10, 6))
        for run_idx, run_accuracies in enumerate(all_accuracies):
            # Usa la variabile temporanea come label per l'ultimo plot
            if run_idx < len(run_labels):
                label = run_labels[run_idx]  # Prendi la label dalla lista
            else:
                label = cfg.name  # Default label per le run oltre la lista

            plt.plot(range(1, len(run_accuracies) + 1), run_accuracies, 
                     marker='o', linestyle='-', label=label)

        plt.title('RANDOM SPLIT WITH 10 CLIENTS')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()"""

    else:
        trainset, testset = load_dataset(
            data_cfg=cfg.dataset,
            num_clients=cfg.num_clients,
            federated=cfg.federated,
            partitioning=cfg.partitioning
        )
        
        num_epochs = 50
        batch_size = cfg.batch_size
        learning_rate = cfg.learning_rate
        momentum = cfg.momentum
        weight_decay = cfg.weight_decay
        
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        model = MulticlassNet(cfg.dataset_path, partitioning=cfg.partitioning)
       
        optimizer = SGD(
            model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )

        train_centralized_multi(model, train_loader, optimizer, num_epochs, device)

        test_loss, test_accuracy, f1_score = test_multi(model, test_loader, device)

        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")
        print(f"F1-score: {f1_score}")


def save_accuracies(accuracies, save_path):
    # Controlla se il file esiste, se sì, carica le vecchie accuracy
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = []  # Crea una nuova lista se il file non esiste
    
    # Aggiungi le nuove accuracy alla lista
    data.append(accuracies)

    # Salva le nuove accuracy
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def load_accuracies(save_path):
    # Carica le accuracy salvate in precedenza
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        return []

if __name__ == "__main__":
    main()