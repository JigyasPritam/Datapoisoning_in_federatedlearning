import flwr as fl
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import json
import os

from model.net import MLP, CNN
from client.flower_client import FlowerClient
from server.strategy import get_strategy

# Config
NUM_CLIENTS = 10
NUM_ROUNDS = 20
BATCH_SIZE = 32
DATASET = "MNIST"  # change to "EMNIST" for EMNIST
MODEL = "MLP"      # change to "CNN" for CNN
NUM_CLASSES = 10   # 10 for MNIST, 62 for EMNIST
SEED = 42
DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")
print(f"Dataset: {DATASET} | Model: {MODEL} | Clients: {NUM_CLIENTS} | Rounds: {NUM_ROUNDS}")

# Load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

if DATASET == "MNIST":
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
elif DATASET == "EMNIST":
    train_ds = datasets.EMNIST(root="./data", split="byclass", train=True, download=True, transform=transform)
    test_ds = datasets.EMNIST(root="./data", split="byclass", train=False, download=True, transform=transform)

# IID partition
rng = np.random.default_rng(SEED)
indices = np.arange(len(train_ds))
rng.shuffle(indices)
client_indices = np.array_split(indices, NUM_CLIENTS)

# Client function
def client_fn(context):
    cid = int(context.node_config["partition-id"])
    client_indices_subset = client_indices[cid].tolist()

    trainloader = DataLoader(
        Subset(train_ds, client_indices_subset),
        batch_size=BATCH_SIZE, shuffle=True
    )
    testloader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = MLP(num_classes=NUM_CLASSES) if MODEL == "MLP" else CNN(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    return FlowerClient(model, trainloader, testloader, DEVICE).to_client()

# Server function
def server_fn(context):
    strategy = get_strategy(min_clients=NUM_CLIENTS)
    config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS)
    return fl.server.ServerAppComponents(strategy=strategy, config=config)

# Run simulation
client_app = fl.client.ClientApp(client_fn=client_fn)
server_app = fl.server.ServerApp(server_fn=server_fn)

history = fl.simulation.run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=NUM_CLIENTS,
    backend_config={"client_resources": {"num_cpus": 1}}
)

# Save results
os.makedirs("evaluation", exist_ok=True)

if history is not None:
    results = {
        "loss": history.losses_distributed,
        "accuracy": history.metrics_distributed_evaluate["accuracy"]
    }
    filename = f"evaluation/{DATASET.lower()}_{MODEL.lower()}_iid_results.json"
    with open(filename, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {filename}")
else:
    print("Warning: history is None, results not saved")