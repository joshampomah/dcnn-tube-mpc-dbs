"""Cross-platform device utilities for PyTorch."""
from __future__ import annotations

from typing import Optional

import torch


def synchronize_device(device: torch.device) -> None:
    """Synchronize the given device to ensure all operations complete."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    """Empty the cache for the given device."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def get_memory_usage(device: torch.device) -> Optional[dict]:
    """Get memory usage statistics for the device."""
    if device.type == "cuda":
        return {
            "allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        }
    elif device.type == "mps":
        try:
            return {
                "allocated_gb": torch.mps.current_allocated_memory() / 1e9,
                "driver_allocated_gb": torch.mps.driver_allocated_memory() / 1e9,
            }
        except AttributeError:
            return None
    return None
