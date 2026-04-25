# Fe-Learn-Datapoisoning
## Federated Learning with Data Poisoning Attacks

A research project implementing a **Federated Learning (FL) system from scratch** to study data poisoning attacks in federated environments.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Current Progress](#current-progress)
- [Project Structure](#project-structure)
- [Datasets & Models](#datasets--models)
- [Federated Learning Setup](#federated-learning-setup)
- [Baseline Results](#baseline-results)
- [Environment Setup](#environment-setup)
- [Running Experiments](#running-experiments)
- [Next Steps](#next-steps)
- [References](#references)

---

## Project Overview

Federated Learning allows multiple clients to collaboratively train a global ML model without sharing raw data. This project implements a complete FL pipeline using **Flower (flwr) framework** with baseline evaluation across multiple datasets, models, and partitioning strategies.

**Goals:**
1. Establish a robust baseline FL system
2. Evaluate on MNIST and EMNIST with MLP and CNN
3. Support both IID and Non-IID distributions
4. Implement data poisoning attacks
5. Develop defense mechanisms

---

## Current Progress

### Module 1: Baseline Federated Learning Setup
**Status: 90% Complete**

#### Completed
- [x] FL Architecture (Client-Server with Flower 1.28.0)
- [x] Datasets: MNIST (60k) and EMNIST (697k)
- [x] Data Partitioning: IID and Non-IID shard-based
- [x] Models: MLP and CNN with configurable classes
- [x] Client Training: SGD (lr=0.01, momentum=0.9)
- [x] Server Aggregation: FedAvg with logging
- [x] Simulation: 10 clients, 20 rounds

#### Evaluation Status
| Config | Status | Result |
|--------|--------|--------|
| MNIST+MLP+IID | Complete | 98.0% |
| MNIST+MLP+NonIID | Ready | Pending |
| MNIST+CNN+IID | Ready | Pending |
| MNIST+CNN+NonIID | Ready | Pending |
| EMNIST+MLP+IID | Ready | Pending |
| EMNIST+MLP+NonIID | Ready | Pending |
| EMNIST+CNN+IID | Ready | Pending |
| EMNIST+CNN+NonIID | Ready | Pending |
| Centralized | Complete | 98.6% |

### Module 2: Data Poisoning Attacks
**Status: Not Started**

---

## Project Structure

```
Fe-Learn-Datapoisoning/
├── data/              # Datasets
├── model/             # MLP and CNN
├── client/            # FlowerClient
├── server/            # FedAvg strategy
├── evaluation/        # Results
├── notebooks/         # Research
├── main.py            # Entry point
└── readme.md          # This file
```

---

## Datasets & Models

### Datasets
| Dataset | Train | Test | Classes |
|---------|-------|------|---------|
| MNIST | 60K | 10K | 10 |
| EMNIST | 697K | 116K | 62 |

Preprocessing: Mean=0.1307, Std=0.3081

### Models
- **MLP**: 784 -> 128 -> 64 -> output (110K params)
- **CNN**: Conv->Conv->FC->output (50K params)

---

## FL Configuration

| Parameter | Value |
|-----------|-------|
| Framework | Flower 1.28.0 |
| Strategy | FedAvg |
| Clients | 10 |
| Rounds | 20 |
| Batch | 32 |
| Epochs | 1 |
| Optimizer | SGD |
| LR | 0.01 |
| Momentum | 0.9 |

### Data Partitioning
- **IID**: Random uniform split (homogeneous)
- **Non-IID**: Shard-based (2 per client, heterogeneous)

---

## Baseline Results

### MNIST + MLP + IID
| Round | Accuracy | Loss |
|-------|----------|------|
| 1 | 91.40% | 0.00910 |
| 5 | 95.99% | 0.00398 |
| 10 | 97.22% | 0.00277 |
| 15 | 97.72% | 0.00231 |
| 20 | 98.00% | 0.00216 |

### Centralized (MNIST + MLP)
| Epochs | Accuracy | Loss |
|--------|----------|------|
| 5 | 97.8% | 0.00201 |
| 10 | 98.6% | 0.00151 |

**Gap**: ~0.6% (federated vs centralized)

---

## Environment Setup

```bash
conda activate ML_AI
pip install torch torchvision flwr "flwr[simulation]" numpy matplotlib jupyter
python main.py
```

---

## Running Experiments

### Config (main.py)
```python
NUM_CLIENTS = 10
NUM_ROUNDS = 20
DATASET = "MNIST"  # or EMNIST
MODEL = "MLP"      # or CNN
NUM_CLASSES = 10   # 62 for EMNIST
```

### Switch to Non-IID
Comment lines 39-43, uncomment lines 45-54

### Output
```
evaluation/{dataset}_{model}_{strategy}_results.json
```

---

## Next Steps

1. Execute remaining 7 baselines
2. Generate comparison plots
3. Implement poisoning attacks
4. Evaluate robustness
5. Develop defenses

---

## Known Issues

| Issue | Status |
|-------|--------|
| run_simulation() None | Fixed |
| Ray warnings | Harmless |
| CUDA instability | Use CPU |

---

## References

- [FedAvg Paper](https://arxiv.org/abs/1602.05629) - McMahan et al., 2017
- [Backdoor FL](https://arxiv.org/abs/1807.00459) - Bagdasaryan et al., 2021
- [Flower Framework](https://flower.ai)
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)

---

**Project**: Fe-Learn-Datapoisoning (8th Semester Final Project)
**Last Updated**: April 2024

*Academic research on data poisoning attacks and defenses in federated learning.*
