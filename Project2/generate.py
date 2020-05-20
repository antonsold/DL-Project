import torch
import math


def generate_set(size):
    input_ = torch.Tensor(size, 2).uniform_(0, 1)
    # computing labels in [0, 1]
    target_single = input_.sub(0.5).pow(2).sum(axis=1).sub(1 / (math.pi * 2)).sign().view(-1, 1)
    # turn the labels into one-hot encoding
    target = torch.cat((target_single.mul(-1), target_single), 1).add(1).div(2)
    return input_, target
