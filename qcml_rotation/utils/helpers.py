"""
Utility functions for reproducibility, configuration, and I/O.
"""

import random
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(
    results: Dict[str, Any],
    output_dir: str = "results",
    prefix: str = "experiment"
) -> str:
    """
    Save experiment results to JSON file with timestamp.

    Returns path to saved file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    filepath = output_path / filename

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return str(filepath)


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AverageMeter:
    """Tracks running average of a metric during training."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
