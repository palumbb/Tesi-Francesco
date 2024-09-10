import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call, instantiate
from hydra.core.hydra_config import HydraConfig
from client import generate_client_fn


@hydra.main(config_path="../conf", config_name="base", version_base=None)

def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    ## DATASET PREPARATION

    train, test = call(cfg.load_dataset)
    data_preparation = call(cfg.prepare_dataset)
    trainloaders, validationloaders, testloader = data_preparation(train=train,test=test)

    ## CLIENTS INSTANTIATION
    #client_fn = generate_client_fn(trainloaders, validationloaders, cfg.model)
    #print("Successful")

if __name__ == "__main__":
    main()