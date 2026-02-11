import torch


def fedavg(state_dicts):
    avg_state = {}

    for key in state_dicts[0].keys():
        avg_state[key] = torch.mean(
            torch.stack([sd[key] for sd in state_dicts]), dim=0
        )

    return avg_state
