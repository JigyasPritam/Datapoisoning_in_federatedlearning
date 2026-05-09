# ─────────────────────────────────────────────────────────────
# NOTE: This Flower-based client implementation was the original
# approach using flwr.client.NumPyClient and Ray simulation.
# It was replaced by a manual sequential FL loop in main.py due
# to the following technical constraints on Windows:
#   1. flwr.simulation.run_simulation returns None on Windows
#      (history object not accessible — results cannot be saved)
#   2. Ray backend causes MemoryError with EMNIST (697k samples)
#      due to parallel client processes exceeding available RAM
# The manual implementation in main.py is mathematically
# equivalent to this Flower approach using the same FedAvg logic.
# Reference: McMahan et al. (2017) arXiv:1602.05629
# ─────────────────────────────────────────────────────────────
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple
import numpy as np


def train(model, dataloader, epochs, device):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def test(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = train(self.model, self.trainloader, epochs=1, device=self.device)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "loss": loss,
            "accuracy": accuracy
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, device=self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}