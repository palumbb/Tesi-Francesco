import hydra
import pandas as pd
import flwr as fl
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call, instantiate
from hydra.core.hydra_config import HydraConfig
from client import generate_client_fn
from server import get_evalulate_fn, get_on_fit_config
import pickle


@hydra.main(config_path="./conf", config_name="base", version_base=None)

def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    ## DATASET PREPARATION

    train, test = call(cfg.load_dataset)
    data_preparation = call(cfg.prepare_dataset)
    trainloaders, validationloaders, testloader = data_preparation(train=train,test=test)

    ## CLIENTS INSTANTIATION
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.model)
    #print("Clients generated")

    strategy = instantiate(
        cfg.strategy, evaluate_fn=get_evalulate_fn(cfg.model, testloader)
    )

    ## START SIMULATION
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.0},
    )



if __name__ == "__main__":
    main()