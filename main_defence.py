import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset, random_split
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
NUM_MALICIOUS      = 3         # number of malicious clients
ATTACK_TYPE        = "targeted" # "targeted" or "random"
SOURCE_LABEL       = 7
TARGET_LABEL       = 1

# Defense config
DEFENSE            = "flip"  # Options:
                                     # "none"
                                     # "fed_median"
                                     # "trimmed_mean"
                                     # "cosine_similarity"
                                     # "norm_clipping"
                                     # "flip"

# Defense hyperparameters
TRIM_FRACTION      = 0.2        # for trimmed_mean — fraction to trim each side
COSINE_THRESHOLD   = 0.0        # for cosine_similarity — min cosine sim to keep client
CLIP_THRESHOLD     = 3.0        # for norm_clipping — max allowed update norm
FLIP_EPOCHS        = 3          # for flip — fine-tuning epochs on clean server data
FLIP_CLEAN_SIZE    = 200        # for flip — number of clean server samples

print(f"Using device: {DEVICE}")
print(f"Dataset: {DATASET} | Model: {MODEL} | Partition: {PARTITION}")
print(f"Attack: {ATTACK_TYPE} | Malicious: {NUM_MALICIOUS}/{NUM_CLIENTS}")
print(f"Defense: {DEFENSE}")

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
# SERVER CLEAN DATASET (for FLIP defense)
# ─────────────────────────────────────────────
server_clean_loader = None
if DEFENSE == "flip":
    server_clean_size = FLIP_CLEAN_SIZE
    remaining_size    = len(test_ds) - server_clean_size
    server_clean_ds, _ = random_split(
        test_ds,
        [server_clean_size, remaining_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    server_clean_loader = DataLoader(
        server_clean_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    print(f"FLIP: Server clean dataset size = {server_clean_size}")

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
# HELPER: GET MODEL UPDATE (delta weights)
# ─────────────────────────────────────────────
def get_update(client_model, global_model):
    """Returns the difference between client and global model weights."""
    update = {}
    for key in global_model.state_dict():
        update[key] = client_model.state_dict()[key].float() - global_model.state_dict()[key].float()
    return update

def flatten_update(update):
    """Flatten all update tensors into a single 1D vector."""
    return torch.cat([v.flatten() for v in update.values()])

def get_update_norm(update):
    """L2 norm of flattened update."""
    return torch.norm(flatten_update(update)).item()

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


def evaluate_asr(model, dataloader, source_label, target_label):
    model.eval()
    source_total, source_as_target = 0, 0
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

# ─────────────────────────────────────────────
# DEFENSE 1: STANDARD FEDAVG (no defense)
# ─────────────────────────────────────────────
def fedavg(global_model, client_models, client_sizes):
    total = sum(client_sizes)
    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = sum(
            client_models[i].state_dict()[key].float() * client_sizes[i] / total
            for i in range(len(client_models))
        )
    global_model.load_state_dict(global_dict)
    return global_model

# ─────────────────────────────────────────────
# DEFENSE 2: FED MEDIAN
# ─────────────────────────────────────────────
def fed_median(global_model, client_models):
    """
    Parameter-wise median aggregation.
    Replaces weighted mean with median — robust to outlier poisoned updates.
    Reference: Yin et al. (2018) — Byzantine-Robust Distributed Learning
    """
    global_dict = global_model.state_dict()
    for key in global_dict:
        stacked = torch.stack([
            client_models[i].state_dict()[key].float()
            for i in range(len(client_models))
        ])
        global_dict[key] = torch.median(stacked, dim=0).values
    global_model.load_state_dict(global_dict)
    return global_model

# ─────────────────────────────────────────────
# DEFENSE 3: TRIMMED MEAN
# ─────────────────────────────────────────────
def trimmed_mean(global_model, client_models, trim_fraction=0.2):
    """
    Parameter-wise trimmed mean aggregation.
    Removes top and bottom trim_fraction of values before averaging.
    More robust than median while retaining more information.
    Reference: Yin et al. (2018)
    """
    n = len(client_models)
    trim_count = max(1, int(n * trim_fraction))

    global_dict = global_model.state_dict()
    for key in global_dict:
        stacked = torch.stack([
            client_models[i].state_dict()[key].float()
            for i in range(n)
        ])
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[trim_count: n - trim_count]
        global_dict[key] = trimmed.mean(dim=0)
    global_model.load_state_dict(global_dict)
    return global_model

# ─────────────────────────────────────────────
# DEFENSE 4: NORM CLIPPING
# ─────────────────────────────────────────────
def norm_clipping(global_model, client_models, client_sizes, clip_threshold=3.0):
    """
    Clips client updates whose L2 norm exceeds a threshold before aggregation.
    Limits the influence of any single client — reduces damage from poisoned updates.
    Reference: Sun et al. (2019) — Can You Really Backdoor Federated Learning?
    """
    updates = [get_update(cm, global_model) for cm in client_models]
    norms   = [get_update_norm(u) for u in updates]

    clipped_models = []
    for i, (client_model, update, norm) in enumerate(zip(client_models, updates, norms)):
        if norm > clip_threshold:
            scale = clip_threshold / norm
            clipped_model = copy.deepcopy(global_model)
            clipped_dict  = clipped_model.state_dict()
            for key in clipped_dict:
                clipped_dict[key] = global_model.state_dict()[key].float() + update[key] * scale
            clipped_model.load_state_dict(clipped_dict)
            clipped_models.append(clipped_model)
        else:
            clipped_models.append(client_model)

    return fedavg(global_model, clipped_models, client_sizes)

# ─────────────────────────────────────────────
# DEFENSE 5: COSINE SIMILARITY FILTERING
# ─────────────────────────────────────────────
def cosine_similarity_filter(global_model, client_models, client_sizes, threshold=0.0):
    """
    Filters clients whose update direction deviates significantly from the majority.
    Computes cosine similarity of each client update against the mean update direction.
    Clients below the threshold are excluded from aggregation.
    Inspired by FedDLFA and FLTrust research directions.
    """
    updates      = [get_update(cm, global_model) for cm in client_models]
    flat_updates = [flatten_update(u) for u in updates]

    # Mean update direction
    mean_update = torch.stack(flat_updates).mean(dim=0)

    kept_models = []
    kept_sizes  = []
    rejected    = []

    for i, (flat_u, client_model, size) in enumerate(zip(flat_updates, client_models, client_sizes)):
        cos_sim = F.cosine_similarity(
            flat_u.unsqueeze(0),
            mean_update.unsqueeze(0)
        ).item()

        if cos_sim >= threshold:
            kept_models.append(client_model)
            kept_sizes.append(size)
        else:
            rejected.append(i)

    if len(rejected) > 0:
        print(f"  Cosine filter rejected clients: {rejected}")

    if len(kept_models) == 0:
        print("  Warning: all clients rejected — falling back to FedAvg")
        return fedavg(global_model, client_models, client_sizes)

    return fedavg(global_model, kept_models, kept_sizes)

# ─────────────────────────────────────────────
# DEFENSE 6: SIMPLIFIED FLIP
# ─────────────────────────────────────────────
def simplified_flip(global_model, client_models, client_sizes, clean_loader, fine_tune_epochs=3):
    """
    Simplified FLIP defense — post-aggregation fine-tuning on server clean data.
    After normal FedAvg, server fine-tunes the global model on a small clean dataset
    to overwrite poisoning patterns introduced by malicious clients.
    Inspired by: Zhang et al. (2023) — FLIP: A Provable Defense Framework
    Note: This is a simplified implementation using fine-tuning as the correction step.
    """
    # Step 1: Normal FedAvg aggregation
    global_model = fedavg(global_model, client_models, client_sizes)

    # Step 2: Server fine-tunes on clean data to erase poisoning
    global_model = train_local(global_model, clean_loader, epochs=fine_tune_epochs)

    return global_model

# ─────────────────────────────────────────────
# AGGREGATION ROUTER
# ─────────────────────────────────────────────
def aggregate(global_model, client_models, client_sizes, defense):
    if defense == "none":
        return fedavg(global_model, client_models, client_sizes)
    elif defense == "fed_median":
        return fed_median(global_model, client_models)
    elif defense == "trimmed_mean":
        return trimmed_mean(global_model, client_models, trim_fraction=TRIM_FRACTION)
    elif defense == "norm_clipping":
        return norm_clipping(global_model, client_models, client_sizes, clip_threshold=CLIP_THRESHOLD)
    elif defense == "cosine_similarity":
        return cosine_similarity_filter(global_model, client_models, client_sizes, threshold=COSINE_THRESHOLD)
    elif defense == "flip":
        return simplified_flip(global_model, client_models, client_sizes, server_clean_loader, fine_tune_epochs=FLIP_EPOCHS)
    else:
        raise ValueError(f"Unknown defense: {defense}")

# ─────────────────────────────────────────────
# LOAD CLEAN BASELINE FOR DROP CALCULATION
# ─────────────────────────────────────────────
clean_baseline_file = f"evaluation/{DATASET.lower()}_{MODEL.lower()}_{PARTITION}_results.json"
clean_baseline_acc  = None
if os.path.exists(clean_baseline_file):
    with open(clean_baseline_file) as f:
        clean_data = json.load(f)
    clean_baseline_acc = clean_data['accuracy'][-1][1]
    print(f"Clean baseline accuracy: {clean_baseline_acc*100:.2f}%")

# Load attack results for comparison
attack_file = f"evaluation/{DATASET.lower()}_{MODEL.lower()}_{PARTITION}_attack_{ATTACK_TYPE}_{NUM_MALICIOUS}.json"
attack_acc  = None
if os.path.exists(attack_file):
    with open(attack_file) as f:
        attack_data = json.load(f)
    attack_acc = attack_data['accuracy'][-1][1]
    print(f"Attack accuracy (no defense): {attack_acc*100:.2f}%")

# ─────────────────────────────────────────────
# FL TRAINING LOOP WITH ATTACK + DEFENSE
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
    "config": {
        "dataset"       : DATASET,
        "model"         : MODEL,
        "partition"     : PARTITION,
        "attack_type"   : ATTACK_TYPE,
        "num_malicious" : NUM_MALICIOUS,
        "defense"       : DEFENSE,
        "trim_fraction" : TRIM_FRACTION,
        "cosine_threshold": COSINE_THRESHOLD,
        "clip_threshold": CLIP_THRESHOLD,
        "flip_epochs"   : FLIP_EPOCHS,
        "flip_clean_size": FLIP_CLEAN_SIZE
    }
}

os.makedirs("evaluation/defense_results", exist_ok=True)
filename = f"evaluation/defense_results/{DATASET.lower()}_{MODEL.lower()}_{PARTITION}_attack_{ATTACK_TYPE}_{NUM_MALICIOUS}_defense_{DEFENSE}.json"

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

    # ── Apply defense during aggregation ──
    global_model = aggregate(global_model, client_models, client_sizes, DEFENSE)

    del client_models
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── Metrics ──
    loss, accuracy = evaluate(global_model, testloader)
    asr = evaluate_asr(global_model, testloader, SOURCE_LABEL, TARGET_LABEL) if ATTACK_TYPE == "targeted" else None
    per_class = evaluate_per_class(global_model, testloader, NUM_CLASSES)
    drop = (clean_baseline_acc - accuracy) if clean_baseline_acc is not None else None

    results["loss"].append([round_num, loss])
    results["accuracy"].append([round_num, accuracy])
    results["asr"].append([round_num, asr])
    results["accuracy_drop"].append([round_num, drop])
    results["per_class_accuracy"].append([round_num, per_class])

    asr_str  = f"ASR: {asr*100:.2f}% | " if asr is not None else ""
    drop_str = f"Drop: {drop*100:+.2f}%" if drop is not None else ""
    print(f"Round {round_num:2d} | Loss: {loss:.6f} | Accuracy: {accuracy*100:.2f}% | {asr_str}{drop_str}")

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*65)
print(f"DEFENSE SUMMARY — {DEFENSE.upper()}")
print("="*65)
print(f"Defense         : {DEFENSE}")
print(f"Attack          : {ATTACK_TYPE} | {NUM_MALICIOUS} malicious clients")
print(f"Clean baseline  : {clean_baseline_acc*100:.2f}%" if clean_baseline_acc else "N/A")
print(f"Attack accuracy : {attack_acc*100:.2f}%" if attack_acc else "N/A")
print(f"Defended accuracy: {results['accuracy'][-1][1]*100:.2f}%")
if ATTACK_TYPE == "targeted" and results['asr'][-1][1] is not None:
    print(f"Defended ASR    : {results['asr'][-1][1]*100:.2f}%")
if clean_baseline_acc and attack_acc:
    attack_damage   = (clean_baseline_acc - attack_acc) * 100
    defense_recovery = (results['accuracy'][-1][1] - attack_acc) * 100
    recovery_rate   = (defense_recovery / attack_damage * 100) if attack_damage > 0 else 0
    print(f"Attack damage   : {attack_damage:.2f}%")
    print(f"Defense recovery: {defense_recovery:.2f}%")
    print(f"Recovery rate   : {recovery_rate:.1f}%")
print(f"Source class acc: {results['per_class_accuracy'][-1][1][SOURCE_LABEL]*100:.2f}%")
print(f"Target class acc: {results['per_class_accuracy'][-1][1][TARGET_LABEL]*100:.2f}%")
print(f"Results saved to: {filename}")
print("="*65)