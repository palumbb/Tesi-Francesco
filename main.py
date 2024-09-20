"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import os
import pickle

import flwr as fl
import hydra
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
from model import train, evaluate
from data import load_dataset
from server_fednova import FedNovaServer
from server_scaffold import ScaffoldServer, gen_evaluate_fn
from strategy import FedNovaStrategy, ScaffoldStrategy
from torch.utils.data import DataLoader


@hydra.main(config_path="conf", config_name="fedprox_base", version_base=None)

def main(cfg: DictConfig) -> None:
    

    # 2. Prepare your dataset
    if cfg.federated:
        trainloaders, valloaders, testloader = load_dataset(
            data_cfg=cfg.dataset,
            num_clients=cfg.num_clients,
            federated=cfg.federated
        )

        # 3. Define your clients
        client_fn = None
        # pylint: disable=protected-access
        if cfg.client_fn._target_ == "client_scaffold.gen_client_fn":
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

        device = cfg.server_device
        evaluate_fn = gen_evaluate_fn(testloader, device=device, model=cfg.model)

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
    else:
        trainset, testset = load_dataset(
            data_cfg=cfg.dataset,
            num_clients=cfg.num_clients,
            federated=cfg.federated
        )
        
        #input_dim = trainset.tensors[0].shape[1]  
        model = cfg.model
        num_epochs = cfg.num_epochs
        batch_size = cfg.batch_size

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)


        optimizer = optimizer(model.parameters())

        train(model, train_loader, optimizer, num_epochs, device)

        test_loss, test_accuracy = evaluate(model, test_loader, device)

        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

    save_path = HydraConfig.get().runtime.output_dir
    print(save_path)

    # 7. Save your results
    with open(os.path.join(save_path, "history.pkl"), "wb") as f_ptr:
        pickle.dump(history, f_ptr)


if __name__ == "__main__":
    main()