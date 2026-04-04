# Fe-Learn-Datapoisoning
## Federated Learning with Data Poisoning Attacks

A research project implementing a Federated Learning (FL) system from scratch, with baseline evaluation across multiple datasets and model architectures. This serves as the foundation for studying data poisoning attacks in federated learning environments.

---

## Project Overview

Federated Learning allows multiple clients to collaboratively train a global machine learning model without sharing raw data. This project implements a complete FL pipeline using the Flower (flwr) framework, evaluating baseline performance before introducing data poisoning attacks in subsequent modules.

---

## Project Structure

```
Fe-Learn-Datapoisoning/
│
├── data/                          # Auto-created by torchvision on first run
│
├── model/
│   ├── net.py                     # MLP and CNN model definitions
│   └── test_model.py              # Model verification script
│
├── client/
│   └── flower_client.py           # Flower NumPyClient implementation
│
├── server/
│   └── strategy.py                # FedAvg strategy configuration
│
├── evaluation/                    # Auto-created — stores JSON results
│   ├── mnist_mlp_iid_results.json
│   ├── mnist_mlp_noniid_results.json
│   ├── emnist_mlp_iid_results.json
│   └── emnist_mlp_noniid_results.json
│
├── notebooks/                     # Jupyter notebooks (in Fed-Learn folder)
│   ├── 01_dataset.ipynb           # MNIST loading and verification
│   ├── 02_emnist.ipynb            # EMNIST loading and verification
│   ├── 03_partitioning.ipynb      # IID and Non-IID data partitioning
│   ├── 04_centralized_baseline.ipynb  # Centralized training baseline
│   └── 05_evaluation.ipynb        # Plots and metrics visualization
│
└── main.py                        # Entry point — runs FL simulation
```

---

## Datasets

| Dataset | Train Samples | Test Samples | Classes | Source |
|---------|--------------|--------------|---------|--------|
| MNIST | 60,000 | 10,000 | 10 (digits) | torchvision |
| EMNIST (byclass) | 697,932 | 116,323 | 62 (digits + letters) | torchvision |

Both datasets use the same normalization:
- Mean: `0.1307`
- Std: `0.3081`

---

## Models

### MLP (Multi-Layer Perceptron)
```
Input (784) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(num_classes)
```

### CNN (Convolutional Neural Network)
```
Conv2d(1,32,3) → ReLU → MaxPool → Conv2d(32,64,3) → ReLU → MaxPool → Linear(128) → Linear(num_classes)
```

Both models accept `num_classes` as a parameter:
- `num_classes=10` for MNIST
- `num_classes=62` for EMNIST

---

## Federated Learning Setup

| Parameter | Value |
|-----------|-------|
| Framework | Flower (flwr) 1.28.0 |
| Strategy | FedAvg |
| Number of Clients | 10 |
| Communication Rounds | 20 |
| Batch Size | 32 |
| Local Epochs | 1 |
| Optimizer | SGD (lr=0.01, momentum=0.9) |
| Device | CPU |

---

## Data Partitioning

### IID (Independent and Identically Distributed)
- Dataset randomly shuffled and split uniformly across clients
- Each client gets equal number of samples from all classes

### Non-IID (Shard-based)
- Dataset sorted by label and divided into shards
- Each client assigned 2 shards — biased toward specific classes
- Simulates realistic federated data heterogeneity

---

## Baseline Results

### MNIST + MLP + IID (20 rounds)

| Round | Accuracy | Loss |
|-------|----------|------|
| 1 | 91.40% | 0.00910 |
| 5 | 95.99% | 0.00398 |
| 10 | 97.22% | 0.00277 |
| 15 | 97.72% | 0.00231 |
| 20 | 97.94% | 0.00216 |

---

## Environment Setup

### Requirements
- Python 3.10.18
- Anaconda (ML_AI environment)

### Installation

```bash
pip install torch torchvision flwr "flwr[simulation]" numpy matplotlib
```

### Running the Simulation

```powershell
# Activate ML_AI environment (Windows)
& "D:\install\anaconda\envs\ML_AI\python.exe" main.py
```

### Configuration

Edit the config block at the top of `main.py`:

```python
NUM_CLIENTS = 10       # Number of federated clients
NUM_ROUNDS = 20        # Communication rounds
BATCH_SIZE = 32        # Training batch size
DATASET = "MNIST"      # "MNIST" or "EMNIST"
MODEL = "MLP"          # "MLP" or "CNN"
NUM_CLASSES = 10       # 10 for MNIST, 62 for EMNIST
```

---

## Module Progress

- [x] **Module 1: Baseline Federated Learning Setup**
  - [x] Step 1: FL Architecture Design
  - [x] Step 2: Dataset Selection and Preprocessing
  - [x] Step 3: Data Partitioning (IID + Non-IID)
  - [x] Step 4: Model Definition (MLP + CNN)
  - [x] Step 5: Client-Side Local Training
  - [x] Step 6: Server-Side FedAvg Aggregation
  - [x] Step 7: Communication Rounds Simulation
  - [ ] Step 8: Full Validation and Evaluation (In Progress)
- [ ] **Module 2: Data Poisoning Attacks** (Upcoming)

---

## Known Issues

- `run_simulation` is deprecated in Flower 1.28 — history object returns `None` on Windows. Results are saved manually from terminal output.
- Ray metrics exporter warnings on Windows are harmless and do not affect simulation results.
- Ray + CUDA instability on Windows — simulation runs on CPU for stability.

---

## References

- [Flower Framework](https://flower.ai)
- [FedAvg Paper — McMahan et al., 2017](https://arxiv.org/abs/1602.05629)
- [LEAF Benchmark](https://leaf.cmu.edu)
- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)