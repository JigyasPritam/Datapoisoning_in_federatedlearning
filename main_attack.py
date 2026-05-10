import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import json
import os
import copy

from model.net import MLP, CNN

# ─────────────────────────────────────────────
# CONFIG — change these between runs
# ─────────────────────────────────────────────
NUM_CLIENTS        = 10
NUM_ROUNDS         = 20
BATCH_SIZE         = 32
DATASET            = "EMNIST"    # "MNIST" or "EMNIST"
MODEL              = "MLP"      # "MLP"   or "CNN"
NUM_CLASSES        = 62         # 10 for MNIST, 62 for EMNIST
PARTITION          = "iid"      # "iid"   or "noniid"
SEED               = 42
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Attack config
NUM_MALICIOUS      = 1         # number of malicious clients (try 1, 3, 5)
ATTACK_TYPE        = "targeted" # "targeted" or "random"
SOURCE_LABEL       = 7          # label to flip FROM (targeted only)
TARGET_LABEL       = 1          # label to flip TO   (targeted only)

# ── Save model Only for key visualization runs ──
# Set True for: targeted_1, targeted_3, targeted_5, cnn_targeted_3
#----------------------------------------------
# NOTE: Saving models for all runs will consume a lot of disk space. Only save for key runs that you want to visualize later.
#Save only for mnist_mlp_iid_attack_targeted_1, mnist_mlp_iid_attack_targeted_3 mnist_mlp_iid_attack_targeted_5
# mnist_cnn_iid_attack_targeted_3 , and also for the EMNIST varients of these runs. 
#----------------------------------------------
SAVE_MODEL         = False

print(f"Using device: {DEVICE}")
print(f"Dataset: {DATASET} | Model: {MODEL} | Partition: {PARTITION} | Clients: {NUM_CLIENTS} | Rounds: {NUM_ROUNDS}")
print(f"Attack: {ATTACK_TYPE} | Malicious clients: {NUM_MALICIOUS}/{NUM_CLIENTS}")

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
        labels = np.array([train_ds.dataset.targets[i] for i in train_ds.indices])
    indices = np.arange(len(train_ds))
    sorted_indices = indices[np.argsort(labels)]
    num_shards = NUM_CLIENTS * 2
    shard_size = len(train_ds) // num_shards
    shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]
    rng = np.random.default_rng(SEED)
    rng.shuffle(shards)
    client_indices = [np.concatenate(shards[i*2:(i+1)*2]) for i in range(NUM_CLIENTS)]

# Malicious client IDs — always first NUM_MALICIOUS clients
malicious_clients = list(range(NUM_MALICIOUS))
print(f"Malicious client IDs: {malicious_clients}")

# ─────────────────────────────────────────────
# POISONED DATASET WRAPPER
# ─────────────────────────────────────────────
class PoisonedDataset(Dataset):
    def __init__(self, subset, attack_type, source_label=7, target_label=1, num_classes=10):
        self.subset       = subset
        self.attack_type  = attack_type
        self.source_label = source_label
        self.target_label = target_label
        self.num_classes  = num_classes

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.attack_type == "targeted":
            if label == self.source_label:
                label = self.target_label
        elif self.attack_type == "random":
            label = np.random.randint(0, self.num_classes)
        return image, label

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
    """Global accuracy and loss."""
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


def evaluate_asr(model, dataloader, source_label, target_label):
    """
    Attack Success Rate (ASR) — targeted attacks only.
    % of source label samples misclassified as target label.
    Formula: ASR = correct_flips / total_source_samples
    """
    model.eval()
    source_total     = 0
    source_as_target = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            mask = labels == source_label
            source_total     += mask.sum().item()
            source_as_target += ((predicted == target_label) & mask).sum().item()
    return source_as_target / source_total if source_total > 0 else 0.0


def evaluate_per_class(model, dataloader, num_classes):
    """
    Per-class accuracy — shows collateral damage of the attack.
    Returns list of accuracy per class.
    """
    model.eval()
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            for c in range(num_classes):
                mask = labels == c
                class_correct[c] += ((predicted == c) & mask).sum().item()
                class_total[c]   += mask.sum().item()
    return [
        class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
        for c in range(num_classes)
    ]


def compute_accuracy_drop(clean_acc, attacked_acc):
    """
    Accuracy Drop (%) — difference between clean and attacked accuracy.
    Formula: Drop = clean_acc - attacked_acc
    """
    return clean_acc - attacked_acc


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
# LOAD CLEAN BASELINE ACCURACY FOR DROP CALC
# ─────────────────────────────────────────────
clean_baseline_file = f"evaluation/{DATASET.lower()}_{MODEL.lower()}_{PARTITION}_results.json"
clean_baseline_acc  = None

if os.path.exists(clean_baseline_file):
    with open(clean_baseline_file) as f:
        clean_data = json.load(f)
    clean_baseline_acc = clean_data['accuracy'][-1][1]
    print(f"Clean baseline accuracy loaded: {clean_baseline_acc*100:.2f}%")
else:
    print(f"Warning: Clean baseline file not found at {clean_baseline_file}. Accuracy drop will not be computed.")

# ─────────────────────────────────────────────
# FL TRAINING LOOP WITH ATTACK
# ─────────────────────────────────────────────
testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

global_model = MLP(num_classes=NUM_CLASSES) if MODEL == "MLP" else CNN(num_classes=NUM_CLASSES)
global_model = global_model.to(DEVICE)

results = {
    "loss"              : [],
    "accuracy"          : [],
    "asr"               : [],
    "accuracy_drop"     : [],
    "per_class_accuracy": [],
    "attack_config"     : {
        "dataset"          : DATASET,
        "model"            : MODEL,
        "partition"        : PARTITION,
        "attack_type"      : ATTACK_TYPE,
        "num_malicious"    : NUM_MALICIOUS,
        "source_label"     : SOURCE_LABEL,
        "target_label"     : TARGET_LABEL,
        "malicious_clients": malicious_clients
    }
}

os.makedirs("evaluation", exist_ok=True)
filename = f"evaluation/{DATASET.lower()}_{MODEL.lower()}_{PARTITION}_attack_{ATTACK_TYPE}_{NUM_MALICIOUS}.json"

for round_num in range(1, NUM_ROUNDS + 1):
    client_models = []
    client_sizes  = []

    for cid in range(NUM_CLIENTS):
        client_idx  = client_indices[cid].tolist()
        base_subset = Subset(train_ds, client_idx)

        if cid in malicious_clients:
            dataset_to_use = PoisonedDataset(
                base_subset,
                attack_type  = ATTACK_TYPE,
                source_label = SOURCE_LABEL,
                target_label = TARGET_LABEL,
                num_classes  = NUM_CLASSES
            )
        else:
            dataset_to_use = base_subset

        trainloader = DataLoader(
            dataset_to_use,
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )

        client_model = copy.deepcopy(global_model)
        client_model = train_local(client_model, trainloader, epochs=1)
        client_models.append(client_model)
        client_sizes.append(len(client_idx))

    global_model = fedavg(global_model, client_models, client_sizes)

    del client_models
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── Metric 1 & 2: Global Accuracy and Loss ──
    loss, accuracy = evaluate(global_model, testloader)
    results["loss"].append([round_num, loss])
    results["accuracy"].append([round_num, accuracy])

    # ── Metric 3: ASR (targeted only) ──
    if ATTACK_TYPE == "targeted":
        asr = evaluate_asr(global_model, testloader, SOURCE_LABEL, TARGET_LABEL)
        results["asr"].append([round_num, asr])
    else:
        results["asr"].append([round_num, None])

    # ── Metric 4: Accuracy Drop ──
    if clean_baseline_acc is not None:
        drop = compute_accuracy_drop(clean_baseline_acc, accuracy)
        results["accuracy_drop"].append([round_num, drop])
    else:
        results["accuracy_drop"].append([round_num, None])

    # ── Metric 5: Per-Class Accuracy ──
    per_class = evaluate_per_class(global_model, testloader, NUM_CLASSES)
    results["per_class_accuracy"].append([round_num, per_class])

    # ── Print ──
    if ATTACK_TYPE == "targeted":
        asr_val  = results["asr"][-1][1]
        drop_val = results["accuracy_drop"][-1][1]
        drop_str = f"{drop_val*100:+.2f}%" if drop_val is not None else "N/A"
        print(f"Round {round_num:2d} | Loss: {loss:.6f} | Accuracy: {accuracy*100:.2f}% | ASR: {asr_val*100:.2f}% | Drop: {drop_str}")
    else:
        drop_val = results["accuracy_drop"][-1][1]
        drop_str = f"{drop_val*100:+.2f}%" if drop_val is not None else "N/A"
        print(f"Round {round_num:2d} | Loss: {loss:.6f} | Accuracy: {accuracy*100:.2f}% | Drop: {drop_str}")

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

print(f"\nDone. Results saved to {filename}")

# ─────────────────────────────────────────────
# SAVE MODEL (only for key visualization runs)
# ─────────────────────────────────────────────
if SAVE_MODEL:
    models_dir = os.path.join("evaluation", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_filename = os.path.join(
        models_dir,
        f"{DATASET.lower()}_{MODEL.lower()}_{PARTITION}_attack_{ATTACK_TYPE}_{NUM_MALICIOUS}_model.pth"
    )
    torch.save(global_model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL ROUND SUMMARY")
print("="*60)
print(f"Global Accuracy     : {results['accuracy'][-1][1]*100:.2f}%")
print(f"Global Loss         : {results['loss'][-1][1]:.6f}")
if ATTACK_TYPE == "targeted":
    print(f"ASR (final round)   : {results['asr'][-1][1]*100:.2f}%")
if results['accuracy_drop'][-1][1] is not None:
    print(f"Accuracy Drop       : {results['accuracy_drop'][-1][1]*100:.2f}%")
print(f"Source class acc    : {results['per_class_accuracy'][-1][1][SOURCE_LABEL]*100:.2f}%")
print(f"Target class acc    : {results['per_class_accuracy'][-1][1][TARGET_LABEL]*100:.2f}%")
print("="*60)