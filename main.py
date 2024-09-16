import hydra
import flwr as fl
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call, instantiate
from hydra.core.hydra_config import HydraConfig
from client import generate_client_fn
from server import get_evalulate_fn, weighted_average
from flwr.common import ndarrays_to_parameters
import torch
from torch.utils.data import DataLoader
from model import BinaryNet, train, evaluate

@hydra.main(config_path="./conf", config_name="base", version_base=None)

def main(cfg: DictConfig):
    federated = cfg.federated
    # print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    num_clients=cfg.num_clients
    lr=cfg.config_fit.lr
    batch_size=cfg.batch_size
    num_epochs=cfg.num_epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = call(cfg.optimizer)

    trainset, testset = call(cfg.load_dataset)

    if federated:

         ## DATASET PARTITION

        data_preparation = call(cfg.partition_dataset)
        trainloaders, validationloaders, testloader = data_preparation(train=trainset,test=testset,num_partitions=num_clients)

        ## CLIENTS INSTANTIATION

        client_fn = generate_client_fn(trainloaders, validationloaders, cfg.model, cfg.optimizer)

        """if cfg.strategy.name == "fednova":
            ndarrays = [
                        layer_param.cpu().numpy()
                        for _, layer_param in instantiate(cfg.model).state_dict().items()
                    ]
            init_parameters = ndarrays_to_parameters(ndarrays)
            extra_args={"init_parameters" : init_parameters}"""

        strategy = instantiate(
            cfg.strategy,
            evaluate_fn=get_evalulate_fn(cfg.model, cfg.optimizer, testloader), 
            evaluate_metrics_aggregation_fn=weighted_average,
            #**extra_args,
        )

        ## START SIMULATION

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0.0},
        )


    else:

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        input_dim = trainset.tensors[0].shape[1]  
        model = BinaryNet(input_dim=input_dim, num_classes=1)

        optimizer = optimizer(model.parameters())

        train(model, train_loader, optimizer, num_epochs, device)

        test_loss, test_accuracy = evaluate(model, test_loader, device)

        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

        
if __name__ == "__main__":
    main()