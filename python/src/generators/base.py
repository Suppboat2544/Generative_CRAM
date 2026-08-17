"""
Base Generator Utilities
========================
Common utilities for all generators.
"""

import torch


def get_device():
    """Get optimal device - supports M4 Mac with MPS."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
