import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flwr.common.parameter import ndarrays_to_parameters

class BinaryNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BinaryNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # For binary classification

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train(model, train_loader, optimizer, num_epochs, device):
    criterion = nn.BCELoss()
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()


def evaluate(model, test_loader, device):
    model.eval()
    model.to(device)
    criterion = nn.BCELoss()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            batch_loss = criterion(outputs, y_batch)
            loss += batch_loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    loss = loss / len(test_loader)
    return loss, accuracy

def model_to_parameters(model):
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)
    print("Extracted model parameters!")
    return parameters