from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from ._tensor import ensure_1d, ensure_2d, infer_device
from .prediction import Prediction


class RFFGP(torch.nn.Module):
    def __init__(self, input_dim: int, num_features: int = 100, lengthscale: float = 1.0, outputscale: float = 1.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_features = int(num_features)
        self.register_buffer("omega", torch.randn(self.num_features, self.input_dim) / float(lengthscale))
        self.register_buffer("phase", torch.rand(self.num_features) * (2.0 * np.pi))
        self.log_lengthscale = torch.nn.Parameter(torch.log(torch.tensor(float(lengthscale))))
        self.log_outputscale = torch.nn.Parameter(torch.log(torch.tensor(float(outputscale))))
        self.log_noise = torch.nn.Parameter(torch.log(torch.tensor(0.1)))
        self.log_weight_prior_var = torch.nn.Parameter(torch.log(torch.tensor(1.0)))
        self.weights = torch.nn.Parameter(torch.randn(self.num_features) * 0.05)
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

    def design_matrix(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_2d(x)
        omega_scaled = self.omega / self.lengthscale
        z = x @ omega_scaled.t() + self.phase
        phi = torch.sqrt(2.0 * self.outputscale / max(self.num_features, 1)) * torch.cos(z)
        return phi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.design_matrix(x)
        return phi @ self.weights + self.bias

    def predict(self, x_test: torch.Tensor, x_train: torch.Tensor | None = None) -> Prediction:
        self.eval()
        with torch.no_grad():
            phi_te = self.design_matrix(x_test)
            mean = phi_te @ self.weights + self.bias
            if x_train is not None:
                phi_tr = self.design_matrix(x_train)
                s_diag = _posterior_diag(phi_tr, self.noise, self.weight_prior_var)
                ep_var = (phi_te.pow(2) @ s_diag).clamp_min(0.0)
                std = torch.sqrt((ep_var + self.noise).clamp_min(1e-12))
            else:
                ep_var = (phi_te.pow(2).sum(dim=1) * 0.01).clamp_min(0.0)
                std = torch.sqrt((ep_var + self.noise).clamp_min(1e-12))
            return Prediction(mean=mean, std=std)


def train_rff_gp(
    model: RFFGP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    iterations: int = 300,
    lr: float = 0.01,
) -> List[float]:
    x_train = ensure_2d(x_train)
    y_train = ensure_1d(y_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []
    model.train()
    for _ in range(int(iterations)):
        optimizer.zero_grad()
        pred = model(x_train)
        residual = y_train - pred
        sigma2 = model.noise
        nll = 0.5 * torch.log(2.0 * np.pi * sigma2) + 0.5 * residual.pow(2) / sigma2
        nll = nll.mean()
        prior = 0.5 * model.weights.pow(2).sum() / model.weight_prior_var
        loss = nll + prior
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    return losses


def run_rff_gp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    num_features: int = 100,
    lengthscale: float = 0.2,
    outputscale: float = 1.0,
    iterations: int = 300,
    lr: float = 0.01,
    device: str | torch.device | None = None,
) -> Tuple[Prediction, List[float], RFFGP]:
    x_train = ensure_2d(x_train)
    y_train = ensure_1d(y_train)
    x_test = ensure_2d(x_test)
    dev = infer_device(x_train, device)
    x_train, y_train, x_test = x_train.to(dev), y_train.to(dev), x_test.to(dev)
    model = RFFGP(
        input_dim=x_train.shape[-1],
        num_features=num_features,
        lengthscale=lengthscale,
        outputscale=outputscale,
    ).to(dev)
    losses = train_rff_gp(model, x_train, y_train, iterations=iterations, lr=lr)
    pred = model.predict(x_test, x_train=x_train)
    return pred, losses, model


def _posterior_diag(phi_train: torch.Tensor, noise_var: torch.Tensor, weight_prior_var: torch.Tensor) -> torch.Tensor:
    diag_phi_t_phi = phi_train.pow(2).sum(dim=0)
    sigma2 = noise_var.clamp_min(1e-8)
    tau2 = weight_prior_var.clamp_min(1e-12)
    return 1.0 / (1.0 / tau2 + diag_phi_t_phi / sigma2)

