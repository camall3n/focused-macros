import torch
import numpy
import random

def reset_seeds(seed=0):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

def get_batch(replay):
    batch = type(replay[0])(*map(lambda x: torch.stack(x, dim=0), zip(*replay)))
    return batch
