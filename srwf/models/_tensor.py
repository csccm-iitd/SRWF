from __future__ import annotations

import torch


def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    return x


def ensure_1d(y: torch.Tensor) -> torch.Tensor:
    y = y.float()
    if y.ndim > 1:
        y = y.squeeze(-1)
    return y


def infer_device(x: torch.Tensor, device: str | torch.device | None) -> torch.device:
    if device is None:
        return x.device
    return torch.device(device)

