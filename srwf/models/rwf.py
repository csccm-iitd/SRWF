from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from ._tensor import ensure_1d, ensure_2d, infer_device
from .prediction import Prediction


class _RWFLevel(torch.nn.Module):
    def __init__(self, feature_count: int, input_dim: int, level: int):
        super().__init__()
        self.translations = torch.nn.Parameter(torch.rand(feature_count, input_dim) * 2.0 - 1.0)
        self.log_scales = torch.nn.Parameter(torch.log(torch.ones(feature_count) * (2.0 ** (-level))))
        self.weights = torch.nn.Parameter(torch.randn(feature_count) * 0.01)
        self.importance = torch.nn.Parameter(torch.tensor(1.0))

    def scales(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_scales), min=1e-6, max=1e6)


class RWFGP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        max_level: int = 4,
        base_features: int = 16,
        lengthscale: float = 1.0,
        outputscale: float = 1.0,
        wavelet_type: str = "morlet",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.max_level = int(max_level)
        self.base_features = int(base_features)
        self.wavelet_type = wavelet_type.lower()

        self.levels = torch.nn.ModuleList()
        total = 0
        for level in range(self.max_level):
            count = self.base_features * (2 ** level)
            self.levels.append(_RWFLevel(count, self.input_dim, level))
            total += count
        self.total_features = total

        self.log_lengthscale = torch.nn.Parameter(torch.log(torch.tensor(float(lengthscale))))
        self.log_outputscale = torch.nn.Parameter(torch.log(torch.tensor(float(outputscale))))
        self.log_noise = torch.nn.Parameter(torch.log(torch.tensor(0.1)))
        self.log_weight_prior_var = torch.nn.Parameter(torch.log(torch.tensor(1.0)))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_lengthscale), min=1e-6, max=1e6)

    @property
    def outputscale(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_outputscale), min=1e-6, max=1e6)

    @property
    def noise(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_noise), min=1e-8, max=1e3)

    @property
    def weight_prior_var(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_weight_prior_var), min=1e-6, max=1e6)

    def morlet_wavelet(self, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * z.pow(2)) * torch.cos(5.0 * z)

    def mexican_hat_wavelet(self, z: torch.Tensor) -> torch.Tensor:
        return (1.0 - z.pow(2)) * torch.exp(-0.5 * z.pow(2))

    def _apply_wavelet(self, z: torch.Tensor) -> torch.Tensor:
        if self.wavelet_type == "morlet":
            value = self.morlet_wavelet(z)
        elif self.wavelet_type == "mexican_hat":
            value = self.mexican_hat_wavelet(z)
        else:
            raise ValueError(f"Unknown wavelet_type={self.wavelet_type}")
        if value.ndim == 1:
            return value
        return torch.prod(value, dim=-1)

    def _level_design(self, level: _RWFLevel, x: torch.Tensor) -> torch.Tensor:
        feats = []
        scales = level.scales()
        gate = torch.sigmoid(level.importance)
        for idx in range(level.translations.shape[0]):
            shift = level.translations[idx]
            scale = scales[idx]
            z = (x - shift) / (scale * self.lengthscale)
            val = self._apply_wavelet(z) / torch.sqrt(scale)
            feats.append(val * gate)
        return torch.stack(feats, dim=1)

    def design_matrix(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_2d(x)
        blocks = [self._level_design(level, x) for level in self.levels]
        phi = torch.cat(blocks, dim=1) if blocks else torch.zeros(x.shape[0], 1, device=x.device)
        phi = torch.sqrt(2.0 * self.outputscale / max(self.total_features, 1)) * phi
        return phi

    def flat_weights(self) -> torch.Tensor:
        return torch.cat([level.weights for level in self.levels], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.design_matrix(x)
        return phi @ self.flat_weights() + self.bias

    def predict(self, x_test: torch.Tensor, x_train: torch.Tensor | None = None) -> Prediction:
        self.eval()
        with torch.no_grad():
            phi_te = self.design_matrix(x_test)
            mean = phi_te @ self.flat_weights() + self.bias
            if x_train is not None:
                phi_tr = self.design_matrix(x_train)
                s_diag = _posterior_diag(phi_tr, self.noise, self.weight_prior_var)
                ep_var = (phi_te.pow(2) @ s_diag).clamp_min(0.0)
                std = torch.sqrt((ep_var + self.noise).clamp_min(1e-12))
            else:
                ep_var = (phi_te.pow(2).sum(dim=1) * 0.01).clamp_min(0.0)
                std = torch.sqrt((ep_var + self.noise).clamp_min(1e-12))
            return Prediction(mean=mean, std=std)


def train_rwf_gp(
    model: RWFGP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    iterations: int = 800,
    lr: float = 0.01,
) -> List[float]:
    x_train = ensure_2d(x_train)
    y_train = ensure_1d(y_train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses: List[float] = []
    model.train()
    for _ in range(int(iterations)):
        optimizer.zero_grad()
        pred = model(x_train)
        residual = y_train - pred
        sigma2 = model.noise
        nll = 0.5 * residual.pow(2).mean() / sigma2 + 0.5 * torch.log(sigma2)
        prior = 0.5 * model.flat_weights().pow(2).sum() / model.weight_prior_var
        loss = nll + prior
        losses.append(float(loss.detach()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    return losses


def run_rwf_gp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    max_level: int = 4,
    base_features: int = 16,
    lengthscale: float = 1.0,
    outputscale: float = 1.0,
    wavelet_type: str = "morlet",
    iterations: int = 800,
    lr: float = 0.01,
    device: str | torch.device | None = None,
) -> Tuple[Prediction, List[float], RWFGP]:
    x_train = ensure_2d(x_train)
    y_train = ensure_1d(y_train)
    x_test = ensure_2d(x_test)
    dev = infer_device(x_train, device)
    x_train, y_train, x_test = x_train.to(dev), y_train.to(dev), x_test.to(dev)
    model = RWFGP(
        input_dim=x_train.shape[-1],
        max_level=max_level,
        base_features=base_features,
        lengthscale=lengthscale,
        outputscale=outputscale,
        wavelet_type=wavelet_type,
    ).to(dev)
    losses = train_rwf_gp(model, x_train, y_train, iterations=iterations, lr=lr)
    pred = model.predict(x_test, x_train=x_train)
    return pred, losses, model


def _posterior_diag(phi_train: torch.Tensor, noise_var: torch.Tensor, weight_prior_var: torch.Tensor) -> torch.Tensor:
    diag_phi_t_phi = phi_train.pow(2).sum(dim=0)
    sigma2 = noise_var.clamp_min(1e-8)
    tau2 = weight_prior_var.clamp_min(1e-12)
    return 1.0 / (1.0 / tau2 + diag_phi_t_phi / sigma2)

