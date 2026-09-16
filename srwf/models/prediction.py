from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Prediction:
    mean: torch.Tensor
    std: torch.Tensor

    def confidence_region(self, n_std: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
        scale = float(n_std)
        return self.mean - scale * self.std, self.mean + scale * self.std

