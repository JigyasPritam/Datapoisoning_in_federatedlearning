import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import json
import os
import copy

from model.net import MLP, CNN

# ─────────────────────────────────────────────
# CONFIG — only change these lines between runs
# ─────────────────────────────────────────────
NUM_CLIENTS = 10
NUM_ROUNDS  = 20
BATCH_SIZE  = 32
DATASET     = "MNIST"   # "MNIST" or "EMNIST"
MODEL       = "CNN"     # "MLP"   or "CNN"
NUM_CLASSES = 62        # 10 for MNIST, 62 for EMNIST
PARTITION   = "noniid"     # "iid"   or "noniid"
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Dataset: {DATASET} | Model: {MODEL} | Partition: {PARTITION} | Clients: {NUM_CLIENTS} | Rounds: {NUM_ROUNDS}")

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

if DATASET == "MNIST":
    train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
elif DATASET == "EMNIST":
    train_ds = datasets.EMNIST(root="./data", split="byclass", train=True,  download=True, transform=transform)
    test_ds  = datasets.EMNIST(root="./data", split="byclass", train=False, download=True, transform=transform)

    # Stratified 60k subset for EMNIST to reduce memory usage
    def stratified_subset(dataset, total_size, seed=42):
        rng_s = np.random.default_rng(seed)
        labels = np.array(dataset.targets)
        classes = np.unique(labels)
        per_class = total_size // len(classes)
        selected = []
        for cls in classes:
            cls_indices = np.where(labels == cls)[0]
            chosen = rng_s.choice(cls_indices, min(per_class, len(cls_indices)), replace=False)
            selected.extend(chosen.tolist())
        return Subset(dataset, selected)

    train_ds = stratified_subset(train_ds, 60000)

# ─────────────────────────────────────────────
# PARTITION DATA
# ─────────────────────────────────────────────
if PARTITION == "iid":
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(train_ds))
    rng.shuffle(indices)
    client_indices = np.array_split(indices, NUM_CLIENTS)

elif PARTITION == "noniid":
    if DATASET == "MNIST":
        labels = np.array(train_ds.targets)
    elif DATASET == "EMNIST":
        # targets not directly available on Subset
        labels = np.array([train_ds.dataset.targets[i] for i in train_ds.indices])
    indices = np.arange(len(train_ds))
    sorted_indices = indices[np.argsort(labels)]
    num_shards = NUM_CLIENTS * 2
    shard_size = len(train_ds) // num_shards
    shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]
    rng = np.random.default_rng(SEED)
    rng.shuffle(shards)
    client_indices = [np.concatenate(shards[i*2:(i+1)*2]) for i in range(NUM_CLIENTS)]

# ─────────────────────────────────────────────
# TRAINING & EVALUATION FUNCTIONS
# ─────────────────────────────────────────────
def train_local(model, dataloader, epochs=1):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model


def evaluate(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def fedavg(global_model, client_models, client_sizes):
    total = sum(client_sizes)
    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = sum(
            client_models[i].state_dict()[key] * client_sizes[i] / total
            for i in range(len(client_models))
        )
    global_model.load_state_dict(global_dict)
    return global_model

# ─────────────────────────────────────────────
# FL TRAINING LOOP
# ─────────────────────────────────────────────
testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

global_model = MLP(num_classes=NUM_CLASSES) if MODEL == "MLP" else CNN(num_classes=NUM_CLASSES)
global_model = global_model.to(DEVICE)

results = {"loss": [], "accuracy": []}
os.makedirs("evaluation", exist_ok=True)
filename = f"evaluation/{DATASET.lower()}_{MODEL.lower()}_{PARTITION}_results.json"

for round_num in range(1, NUM_ROUNDS + 1):
    client_models = []
    client_sizes  = []

    for cid in range(NUM_CLIENTS):
        client_idx = client_indices[cid].tolist()
        trainloader = DataLoader(
            Subset(train_ds, client_idx),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        client_model = copy.deepcopy(global_model)
        client_model = train_local(client_model, trainloader, epochs=1)
        client_models.append(client_model)
        client_sizes.append(len(client_idx))

    global_model = fedavg(global_model, client_models, client_sizes)

    # Free memory
    del client_models
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    loss, accuracy = evaluate(global_model, testloader)
    results["loss"].append([round_num, loss])
    results["accuracy"].append([round_num, accuracy])

    print(f"Round {round_num:2d} | Loss: {loss:.6f} | Accuracy: {accuracy*100:.2f}%")

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

print(f"Done. Results saved to {filename}")