from collections import OrderedDict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from model.binarynet import BinaryNet, test_binary
from model.multiclassnet import MulticlassNet, test_multi


def get_on_fit_config(config: DictConfig):

    def fit_config_fn(server_round: int):
        return {
            "lr": config.lr,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn

def get_evaluate_fn(model_cfg, optimizer_cfg, testloader):

    def evaluate_fn(server_round: int, parameters, config):
        model = instantiate(model_cfg)
        optimizer = instantiate(optimizer_cfg)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        loss, accuracy = test_multi(model, testloader, device)
        return loss, {"global accuracy": accuracy}

    return evaluate_fn

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}
