from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.parameter import ndarrays_to_parameters
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.nn.parameter import Parameter
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class MulticlassNet(nn.Module):
    def __init__(self, data, partitioning, num_classes=5):  # Impostiamo num_classes=4
        super(MulticlassNet, self).__init__()
        if data == "./data/car.csv":
            input_dim = 21
        elif data == "./data/nursery.csv":
            input_dim = 27
        if data=="./data/consumer.csv":
            input_dim = 16
        elif data == "./data/mv.csv":
            input_dim = 14
        if data=="./data/shuttle.csv":
            input_dim = 9
        elif data == "./data/wall-robot-navigation.csv":
            input_dim = 4
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out) 
        return out

def train_centralized_multi(model, train_loader, optimizer, num_epochs, device):
    criterion = nn.CrossEntropyLoss()  # CrossEntropy per multi-class
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            # y_batch = y_batch.long()
            # y_batch = torch.argmax(y_batch, dim=1)
            loss = criterion(outputs, y_batch)  # y_batch è l'indice della classe
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
        #target = target.long()
        #print(target)
        
        # IL PROBLEMA E' CHE TORCH.ARGMAX TRASFORMA IL TARGET IN UN TENSORE DI TUTTI 0
        #print(target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return net

"""def test_multi(
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

            target = target.float()

            output = net(data)
            print(output)
            # classe con probabilità più alta
            _, predicted = torch.max(output, 1)
            target = torch.argmax(target, dim=1)

            #print(predicted)
            #print(target)

            loss += criterion(output, target).item()  # CrossEntropy lavora con target long
          
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    loss = loss / total
    acc = correct / total

    f1 = f1_score(all_targets, all_predictions, average='macro')  # Macro per classi bilanciate

    return loss, acc, f1"""

def test_multi(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float, float]:
    """Evaluate the network on the test set."""
    criterion = nn.CrossEntropyLoss()
    net.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            predicted = torch.argmax(output, dim=1) 
            
            loss = criterion(output, target)
            target = target.long()
            target = torch.argmax(target, dim=1)
            total_loss += loss.item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calcolo della loss media
    avg_loss = total_loss / len(testloader)
    # Calcolo dell'accuracy
    acc = accuracy_score(all_targets, all_predictions)
    
    # Calcolo della F1-score (weighted per bilanciare tra tutte le classi)
    f1 = f1_score(all_targets, all_predictions, average='weighted')

    return avg_loss, acc, f1