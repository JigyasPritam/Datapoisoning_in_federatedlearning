import flwr as fl
import json
import os
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total}


class FedAvgWithLogging(fl.server.strategy.FedAvg):
    def __init__(self, filename, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.losses = []
        self.accuracies = []

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is not None:
            loss, metrics = aggregated
            accuracy = metrics.get("accuracy", None)
            self.losses.append([server_round, loss])
            if accuracy is not None:
                self.accuracies.append([server_round, accuracy])
            self._save()
        return aggregated

    def _save(self):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        results = {
            "loss": self.losses,
            "accuracy": self.accuracies
        }
        with open(self.filename, "w") as f:
            json.dump(results, f, indent=2)


def get_strategy(min_clients=5, filename="evaluation/results.json"):
    strategy = FedAvgWithLogging(
        filename=filename,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    return strategy