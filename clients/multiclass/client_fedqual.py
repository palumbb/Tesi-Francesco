"""Defines the client class and support functions for FedAvg."""

from typing import Callable, Dict, List, OrderedDict

import numpy as np
import torch
from flwr.common import Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from flwr.common import Context
from flwr.client import NumPyClient

from model.multiclassnet import test_multi, train_fedavg

# pylint: disable=too-many-instance-attributes
class FlowerClientFedQual(NumPyClient):
    """Flower client implementing FedQual."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        dirty_percentage: float,
        SE: float,
        N_tot: int,
        beta: float,
        gamma: float,
        delta: float
    ) -> None:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.SE = SE
        self.dirty_percentage = dirty_percentage
        self.N_tot = N_tot
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for FedAvg with custom weights."""
        # Set global model parameters
        self.set_parameters(parameters)


        # Train the model locally
        train_fedavg(
            self.net,
            self.trainloader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
        )

        # Retrieve updated model parameters
        final_p_np = self.get_parameters({})

        # COMPUTE QUALITY METRICS THAT WE WANT TO TAKE INTO ACCOUNT
        C = 1 - self.dirty_percentage # COMPLETENESS
        N_i = len(self.trainloader.dataset) # DIMENSIONALITY
        D = N_i*C/self.N_tot # COMPLETENESS AND DIMENSIONALITY

        # FINAL FORMULA FOR THE QUALITY WEIGHT


        # 1 
        # quality_weight =  self.beta*D + self.gamma*self.SE 

        # 2
        quality_weight = (N_i/self.N_tot)*(self.beta*C + self.gamma*self.SE)

        # 3
        quality_weight = self.delta*N_i/self.N_tot + self.beta*C + self.gamma*self.SE



        #print(f"quality weight: {quality_weight} ")

        # Return updated model parameters, number of samples, and custom metrics
        return final_p_np, N_i, {"quality_weight": quality_weight}


    def evaluate(self, parameters, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, acc, f1 = test_multi(self.net, self.valloader, self.device)  # Usa test_multiclass
        return float(loss), len(self.valloader.dataset), {
            "accuracy": float(acc),
            "f1-score": float(f1),  # Aggiungi F1-score per valutazioni più dettagliate
        }



# pylint: disable=too-many-arguments
def gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    num_epochs: int,
    learning_rate: float,
    model: DictConfig,
    quality_metrics: float,
    N_tot: int,
    beta: float,
    gamma: float,
    delta: float,
    momentum: float,
    weight_decay: float,
) -> Callable[[Context], FlowerClientFedQual]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the FedQual flower clients.

    Parameters
    ----------
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    momentum : float
        The momentum for SGD optimizer of clients
    weight_decay : float
        The weight decay for SGD optimizer of clients

    Returns
    -------
    Callable[[Context], FlowerClientFedAvg]
        The client function that creates the FedAvg flower clients
    """

    def client_fn(context: Context) -> FlowerClientFedQual:
        """Create a Flower client representing a single organization."""
        # Access client ID from the context

        cid = int(context.node_config["partition-id"])


        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        # Load data specific to the client
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        dirty_percentage = quality_metrics[int(cid)][0]
        SE = quality_metrics[cid][1]  # Already normalized
        
        # Create and return the client
        return NumPyClient.to_client(FlowerClientFedQual(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            momentum,
            weight_decay,
            dirty_percentage,
            SE,
            N_tot,
            beta,
            gamma,
            delta,
        ) )
    return client_fn