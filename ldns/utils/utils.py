import os
import random
import numpy as np
import torch


def count_parameters(model):
    """
    Count number of trainable parameters in model.

    Args:
        model: Pytorch model.

    Returns:
        Number of trainable parameters.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int):
    """
    Set the seed for all random number generators and switch to deterministic algorithms.
    This can hurt performance!

    Args:
        seed: The random seed.
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

