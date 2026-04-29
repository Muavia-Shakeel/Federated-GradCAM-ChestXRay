"""
FedAvg: dataset-size-weighted averaging of client model parameters.
"""
import copy
import torch
import torch.nn as nn


def fedavg(
    global_model: nn.Module,
    client_models: list[nn.Module],
    client_sizes: list[int],
) -> nn.Module:
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]

    global_state = copy.deepcopy(client_models[0].state_dict())

    for key in global_state:
        global_state[key] = sum(
            weights[i] * client_models[i].state_dict()[key].float()
            for i in range(len(client_models))
        )

    global_model.load_state_dict(global_state)
    return global_model


def broadcast_weights(global_model: nn.Module, client_models: list[nn.Module]):
    """Copy global weights to all clients."""
    global_state = global_model.state_dict()
    for model in client_models:
        model.load_state_dict(copy.deepcopy(global_state))
