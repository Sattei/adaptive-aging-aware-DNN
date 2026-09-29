"""Small model-inspection helpers shared by experiment controls."""

import torch.nn as nn


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters that will receive gradients."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
