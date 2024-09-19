from collections import OrderedDict
from typing import Dict, Tuple

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate

from model import evaluate, train_fedprox


class FedProxFlowerClient(fl.client.NumPyClient):

    def __init__(self, trainloader, valloader, model_cfg, optimizer_cfg) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.model = instantiate(model_cfg)
        self.optimizer = instantiate(optimizer_cfg)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True) 

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        lr = config["lr"]
        epochs = config["epochs"]
        proximal_term = config["proximal_term"]
        optimizer = self.optimizer(self.model.parameters())

        #can set this via a config
        train_fedprox(self.model, self.trainloader, optimizer, epochs, proximal_term, self.device)
        return self.get_parameters({}), len(self.trainloader), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = evaluate(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {"local accuracy": accuracy}
    
def generate_client_fn(trainloaders, valloaders, model_cfg, optimizer_cfg):

    def client_fn(cid: str):
        return FedProxFlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            model_cfg=model_cfg,
            optimizer_cfg=optimizer_cfg,
        )

    return client_fn