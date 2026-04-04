import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total}


def get_strategy(min_clients=5):
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    return strategy