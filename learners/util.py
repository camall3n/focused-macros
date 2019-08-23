import torch

def get_batch(replay):
    batch = type(replay[0])(*map(lambda x: torch.stack(x, dim=0), zip(*replay)))
    return batch
