"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import os
import pickle
import matplotlib.pyplot as plt
import flwr as fl
import hydra
import random
import torch
import numpy as np
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
from model.binarynet import BinaryNet, train_centralized_binary, test_binary
from model.multiclassnet import MulticlassNet, train_centralized_multi, test_multi
from data_handler.data import load_dataset
from servers.server_fednova import FedNovaServer
from servers.server_fedqual import FedQualServer
from servers.server_scaffold import ScaffoldServer, gen_evaluate_fn
from strategy import FedNovaStrategy, ScaffoldStrategy, FedQualStrategy
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
import pandas as pd

@hydra.main(config_path="conf", config_name="fedqual_base", version_base=None)
def main(cfg: DictConfig) -> None:
    
    seed = 205
    set_seed(seed)

    device = cfg.server_device

    print(f"Strategy: {cfg.name}")
    print("Dataset: " + str(cfg.dataset_path))
    print("Partitioning: " + str(cfg.partitioning))
    print("Quality: " + str(cfg.quality))
    print("Imputation: " + str(cfg.imputation))
    print(f"Dirty Percentage: {cfg.dirty_percentage}")
    #print("Clients: " + str(cfg.num_clients))
    #print("Local epochs: " + str(cfg.num_epochs))
    #print("Sampled clients: " + str(cfg.clients_per_round))
    #print("Rounds: " + str(cfg.num_rounds))

    accuracies = []
    run_labels = ["FedAvg", "FedProx", "FedNova", "Scaffold"]

    # 2. Preparazione del dataset
    if cfg.federated:
        trainloaders, valloaders, testloader, quality_metrics, N_tot = load_dataset(
            data_cfg=cfg.dataset,
            num_clients=cfg.num_clients,
            federated=cfg.federated,
            partitioning=cfg.partitioning,
            model=cfg.model,
            dirty_percentage=cfg.dirty_percentage,
            quality=cfg.quality,
            imputation=cfg.imputation,
            seed=seed
        )

        # 3. Definizione dei client
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
        elif cfg.strategy._target_ == "strategy.FedQualStrategy":
            target = call(cfg.client_fn)
            client_fn = target(
                trainloaders,
                valloaders,
                model=cfg.model,
                quality_metrics=quality_metrics,
                N_tot=N_tot,
                beta = cfg.client_fn.beta,
                gamma = cfg.client_fn.gamma
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

        # 4. Definizione della strategia
        strategy = instantiate(
            cfg.strategy,
            evaluate_fn=evaluate_fn,
        )

        # 5. Definizione del server
        server = Server(strategy=strategy, client_manager=SimpleClientManager())
        if isinstance(strategy, FedNovaStrategy):
            server = FedNovaServer(strategy=strategy, client_manager=SimpleClientManager())
        elif isinstance(strategy, ScaffoldStrategy):
            server = ScaffoldServer(
                strategy=strategy, model=cfg.model, client_manager=SimpleClientManager()
            )
        elif isinstance(strategy, FedQualStrategy):
            server = FedQualServer(
                strategy=strategy, client_manager=SimpleClientManager()
            )

        # 6. Avvio della simulazione
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

        # 7. Salvataggio dei risultati
        with open(os.path.join(save_path, "history.pkl"), "wb") as f_ptr:
            pickle.dump(history, f_ptr)

    else:
        trainset, testset = load_dataset(
            data_cfg=cfg.dataset,
            num_clients=cfg.num_clients,
            federated=cfg.federated,
            partitioning=cfg.partitioning,
            model=cfg.model,
            quality=cfg.quality,
            dirty_percentage=cfg.dirty_percentage,
            imputation=cfg.imputation,
            seed=seed
        )
        
        num_epochs = 50
        batch_size = cfg.batch_size
        learning_rate = cfg.learning_rate
        momentum = cfg.momentum
        weight_decay = cfg.weight_decay
        
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        model = MulticlassNet(cfg.dataset_path, imputation=cfg.imputation)
       
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
    
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    main()