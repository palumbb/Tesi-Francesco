from collections import OrderedDict
from typing import Dict, Tuple

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate

from model import evaluate, train_scaffold
import os

class ScaffoldFlowerClient(fl.client.NumPyClient):

    def __init__(self, trainloader, valloader, save_dir, model_cfg, optimizer_cfg) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.model = instantiate(model_cfg)
        self.optimizer = instantiate(optimizer_cfg)
        self.lr = optimizer_cfg["lr"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.client_cv = []
        for param in self.model.parameters():
            self.client_cv.append(torch.zeros(param.shape))

        if save_dir == "":
            save_dir = "client_cvs"
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True) 

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        # copy parameters sent by the server into client's local model
        
        server_cv = parameters[len(parameters)//2:]
        parameters = parameters[: len(parameters)//2]
        self.set_parameters(parameters)

        self.client_cv = []
        for param in self.model.parameters():
            self.client_cv.append(param.clone().detach())
        # load client control variate
        if os.path.exists(f"{self.dir}/client_cv_{self.cid}.pt"):
            self.client_cv = torch.load(f"{self.dir}/client_cv_{self.cid}.pt")

        server_cv = [torch.Tensor(cv) for cv in server_cv]
        epochs = config["epochs"]

        optimizer = self.optimizer(self.model.parameters())

        #can set this via a config
        train_scaffold(self.model, self.trainloader, optimizer, epochs, self.device)
        x = parameters
        y_i = self.get_parameters(config={})
        c_i_n = []
        server_update_x = []
        server_update_c = []
        # update client control variate c_i_1 = c_i - c + 1/eta*K (x - y_i)
        for c_i_j, c_j, x_j, y_i_j in zip(self.client_cv, server_cv, x, y_i):
            c_i_n.append(
                c_i_j
                - c_j
                + (1.0 / (self.learning_rate * self.num_epochs * len(self.trainloader)))
                * (x_j - y_i_j)
            )
            # y_i - x, c_i_n - c_i for the server
            server_update_x.append((y_i_j - x_j))
            server_update_c.append((c_i_n[-1] - c_i_j).cpu().numpy())
        self.client_cv = c_i_n
        torch.save(self.client_cv, f"{self.dir}/client_cv_{self.cid}.pt")

        combined_updates = server_update_x + server_update_c

        return (
            combined_updates,
            len(self.trainloader.dataset),
            {},
        )
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = evaluate(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {"local accuracy": accuracy}

def generate_client_fn(trainloaders, valloaders, client_cv_dir, model_cfg, optimizer_cfg):

    def client_fn(cid: str):
        return ScaffoldFlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            model_cfg=model_cfg,
            optimizer_cfg=optimizer_cfg,
            save_dir=client_cv_dir
        )

    return client_fn