from __future__ import annotations

from typing import List, Tuple

import gpytorch
import torch
from gpytorch.models import ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from ._tensor import ensure_1d, ensure_2d, infer_device
from .prediction import Prediction


class ExactMaternGP(ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class SparseMaternSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        q = CholeskyVariationalDistribution(inducing_points.size(0))
        strategy = VariationalStrategy(self, inducing_points, q, learn_inducing_locations=True)
        super().__init__(strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=inducing_points.shape[-1])
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def train_exact_gp(
    model: ExactMaternGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    iterations: int = 180,
    lr: float = 0.01,
) -> List[float]:
    x_train = ensure_2d(x_train)
    y_train = ensure_1d(y_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    losses: List[float] = []
    model.train()
    likelihood.train()
    for _ in range(int(iterations)):
        optimizer.zero_grad()
        out = model(x_train)
        loss = -mll(out, y_train)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    return losses


def train_svgp(
    model: SparseMaternSVGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    iterations: int = 500,
    lr: float = 0.04,
) -> List[float]:
    x_train = ensure_2d(x_train)
    y_train = ensure_1d(y_train)
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=lr,
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=x_train.size(0))
    losses: List[float] = []
    model.train()
    likelihood.train()
    for _ in range(int(iterations)):
        optimizer.zero_grad()
        out = model(x_train)
        loss = -mll(out, y_train)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    return losses


def run_exact_gp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    iterations: int = 180,
    lr: float = 0.01,
    device: str | torch.device | None = None,
) -> Tuple[Prediction, List[float], ExactMaternGP, gpytorch.likelihoods.GaussianLikelihood]:
    x_train = ensure_2d(x_train)
    y_train = ensure_1d(y_train)
    x_test = ensure_2d(x_test)
    dev = infer_device(x_train, device)
    x_train, y_train, x_test = x_train.to(dev), y_train.to(dev), x_test.to(dev)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dev)
    model = ExactMaternGP(x_train, y_train, likelihood).to(dev)
    losses = train_exact_gp(model, likelihood, x_train, y_train, iterations=iterations, lr=lr)
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred = likelihood(model(x_test))
        lower, upper = pred.confidence_region()
        std = ((upper - lower) / 4.0).clamp_min(1e-12)
        out = Prediction(mean=pred.mean, std=std)
    return out, losses, model, likelihood


def run_svgp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    inducing_count: int = 60,
    iterations: int = 500,
    lr: float = 0.04,
    device: str | torch.device | None = None,
) -> Tuple[Prediction, List[float], SparseMaternSVGP, gpytorch.likelihoods.GaussianLikelihood]:
    x_train = ensure_2d(x_train)
    y_train = ensure_1d(y_train)
    x_test = ensure_2d(x_test)
    dev = infer_device(x_train, device)
    x_train, y_train, x_test = x_train.to(dev), y_train.to(dev), x_test.to(dev)
    inducing_points = x_train[: min(int(inducing_count), x_train.size(0)), :].clone()
    model = SparseMaternSVGP(inducing_points).to(dev)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dev)
    losses = train_svgp(model, likelihood, x_train, y_train, iterations=iterations, lr=lr)
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred = likelihood(model(x_test))
        lower, upper = pred.confidence_region()
        std = ((upper - lower) / 4.0).clamp_min(1e-12)
        out = Prediction(mean=pred.mean, std=std)
    return out, losses, model, likelihood

