from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.parameter import ndarrays_to_parameters
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.nn.parameter import Parameter
from sklearn.metrics import f1_score

class MulticlassNet(nn.Module):
    def __init__(self, data, partitioning, num_classes=4):  # Impostiamo num_classes=4
        super(MulticlassNet, self).__init__()
        if data == "./data/consumer.csv":
            input_dim = 16
        elif data == "./data/mv.csv":
            input_dim = 14
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)  # Cambiamo l'output per supportare 4 classi
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Usiamo softmax per la classificazione multi-classe

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)  # Softmax per ottenere probabilità
        return x
    
def train_centralized_multi(model, train_loader, optimizer, num_epochs, device):
    criterion = nn.CrossEntropyLoss()  # CrossEntropy per multi-class
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.long())  # y_batch è l'indice della classe
            loss.backward()
            optimizer.step()

def train_fedavg(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> None:
    criterion = nn.CrossEntropyLoss()  # CrossEntropy per multi-class
    optimizer = SGD(
        net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    net.train()
    for _ in range(epochs):
        net = _train_one_epoch(net, trainloader, device, criterion, optimizer)

def _train_one_epoch(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optimizer,
) -> nn.Module:
    """Train the network on the training set for one epoch."""
    for data, target in trainloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return net

def test_multi(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float, float]:
    criterion = nn.CrossEntropyLoss()  # CrossEntropy per multi-class
    net.eval()
    correct, total, loss = 0, 0, 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            
            loss += criterion(output, target.long()).item()  # CrossEntropy lavora con target long
            
            # Predizioni: prendiamo la classe con la probabilità più alta
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # Calcolo della loss e dell'accuracy
    loss = loss / total
    acc = correct / total
    
    # Calcolo della F1-score per multi-class
    f1 = f1_score(all_targets, all_predictions, average='macro')  # Macro per classi bilanciate

    return loss, acc, f1
