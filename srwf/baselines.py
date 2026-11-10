"""Reference Gaussian process baselines (requires ``gpytorch``)."""

from __future__ import annotations

import time

import gpytorch
import numpy as np
import torch

from .model import Prediction


class GibbsKernel(gpytorch.kernels.Kernel):
    """Gibbs (1997) kernel with input-dependent lengthscale (Paciorek & Schervish, 2004).

    ``l(x) = softplus(b + sum_c w_c exp(-|x - z_c|^2 / 2 h^2))`` is a radial-basis network whose
    centres ``z_c``, width ``h``, weights and bias are learned with the other hyperparameters.
    """

    def __init__(self, centres: torch.Tensor, lengthscale: float = 0.3, **kwargs) -> None:
        super().__init__(**kwargs)
        centres = centres.reshape(len(centres), -1).detach().clone()
        nn_dist = torch.cdist(centres, centres).fill_diagonal_(float("inf")).min(1).values.mean()
        self.centres = torch.nn.Parameter(centres)
        self.raw_width = torch.nn.Parameter(torch.log(2 * nn_dist.clamp_min(1e-3)))
        self.weights = torch.nn.Parameter(torch.zeros(len(centres), dtype=centres.dtype))
        self.bias = torch.nn.Parameter(torch.tensor(float(np.log(np.expm1(lengthscale))), dtype=centres.dtype))

    def lengthscale_at(self, x: torch.Tensor) -> torch.Tensor:
        r2 = (x.unsqueeze(-2) - self.centres.to(x)).pow(2).sum(-1)
        phi = torch.exp(-0.5 * r2 / torch.exp(2 * self.raw_width.to(x)))
        return torch.nn.functional.softplus(phi @ self.weights.to(x) + self.bias.to(x)) + 1e-4

    def forward(self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False, **params):
        if diag:
            return torch.ones(x1.shape[:-1], dtype=x1.dtype, device=x1.device)
        dtype, x1, x2 = x1.dtype, x1.double(), x2.double()
        l1, l2 = self.lengthscale_at(x1).unsqueeze(-1), self.lengthscale_at(x2).unsqueeze(-2)
        s = l1.pow(2) + l2.pow(2)
        r2 = self.covar_dist(x1, x2, square_dist=True)
        return ((2 * l1 * l2 / s).pow(x1.shape[-1] / 2) * torch.exp(-r2 / s)).to(dtype)


class DeepKernel(gpytorch.kernels.Kernel):
    """Deep kernel learning (Wilson et al., 2016): an RBF kernel on the output of an MLP."""

    def __init__(self, d: int, hidden: tuple[int, ...] = (64, 64), features: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        layers, width = [], d
        for h in hidden:
            layers += [torch.nn.Linear(width, h), torch.nn.ReLU()]
            width = h
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(width, features))
        self.base = gpytorch.kernels.RBFKernel(ard_num_dims=features)

    def forward(self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False, **params):
        return self.base.forward(self.net(x1), self.net(x2), diag=diag, **params)


def _kernel(name: str, x: torch.Tensor, **kwargs) -> gpytorch.kernels.Kernel:
    d = x.shape[1]
    if name == "rbf":
        base = gpytorch.kernels.RBFKernel(ard_num_dims=d)
    elif name.startswith("matern"):
        base = gpytorch.kernels.MaternKernel(nu=float(name[-3:]) if "." in name else 2.5, ard_num_dims=d)
    elif name == "gibbs":
        centres = kwargs.pop("centres", 24)
        base = GibbsKernel(x[torch.randperm(len(x))[: min(centres, len(x))]].float().cpu(), **kwargs)
    elif name == "dkl":
        base = DeepKernel(d, **kwargs)
    else:
        raise ValueError(name)
    return gpytorch.kernels.ScaleKernel(base)


class _Standardizer:
    def fit(self, x: np.ndarray, y: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32):
        x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
        self.device, self.dtype = device, dtype
        self.x_mu, self.x_sd = x.mean(0), x.std(0) + 1e-8
        self.y_mu, self.y_sd = float(np.mean(y)), float(np.std(y) + 1e-12)
        return self.tx(x), self.ty(y)

    def tx(self, x) -> torch.Tensor:
        x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
        return torch.as_tensor((x - self.x_mu) / self.x_sd, dtype=self.dtype, device=self.device)

    def ty(self, y) -> torch.Tensor:
        return torch.as_tensor(
            (np.asarray(y, dtype=np.float64) - self.y_mu) / self.y_sd, dtype=self.dtype, device=self.device
        )

    def out(self, mean: torch.Tensor, var: torch.Tensor, noise: float) -> Prediction:
        m = mean.double().cpu().numpy() * self.y_sd + self.y_mu
        v = var.double().cpu().numpy()
        return Prediction(m, np.sqrt(v + noise) * self.y_sd, np.sqrt(np.maximum(v, 0)) * self.y_sd)


class _ExactModel(gpytorch.models.ExactGP):
    def __init__(self, x, y, lik, kernel):
        super().__init__(x, y, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class ExactGP:
    """Exact GP regression, hyperparameters by marginal likelihood (Adam)."""

    def __init__(
        self,
        kernel: str = "matern2.5",
        steps: int = 200,
        lr: float = 0.05,
        kernel_kwargs: dict | None = None,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        self.kernel, self.steps, self.lr = kernel, steps, lr
        self.kernel_kwargs, self.seed = dict(kernel_kwargs or {}), seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seconds = 0.0

    def fit(self, x, y) -> ExactGP:
        t0 = time.perf_counter()
        torch.manual_seed(self.seed)
        self.std = _Standardizer()
        xs, ys = self.std.fit(x, y, self.device, torch.float64)
        self.lik = gpytorch.likelihoods.GaussianLikelihood().to(self.device, torch.float64)
        kernel = _kernel(self.kernel, xs, **self.kernel_kwargs)
        self.model = _ExactModel(xs, ys, self.lik, kernel).to(self.device, torch.float64)
        self.model.train(), self.lik.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.lik, self.model)
        with gpytorch.settings.max_cholesky_size(20000):
            for _ in range(self.steps):
                opt.zero_grad()
                loss = -mll(self.model(xs), ys)
                loss.backward()
                opt.step()
        self.seconds = time.perf_counter() - t0
        return self

    @torch.no_grad()
    def predict(self, x) -> Prediction:
        self.model.eval(), self.lik.eval()
        means, variances = [], []
        xs = self.std.tx(x)
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cholesky_size(20000):
            for chunk in torch.split(xs, 4096):
                f = self.model(chunk)
                means.append(f.mean)
                variances.append(f.variance)
        return self.std.out(torch.cat(means), torch.cat(variances), float(self.lik.noise))


class _SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing, kernel):
        q = gpytorch.variational.CholeskyVariationalDistribution(inducing.size(0))
        s = gpytorch.variational.VariationalStrategy(self, inducing, q, learn_inducing_locations=True)
        super().__init__(s)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class SVGP:
    """Stochastic variational GP (Hensman et al., 2013) with minibatch training."""

    def __init__(
        self,
        num_inducing: int = 512,
        kernel: str = "matern2.5",
        epochs: int = 100,
        batch_size: int = 1024,
        lr: float = 0.01,
        kernel_kwargs: dict | None = None,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        self.num_inducing, self.kernel, self.epochs = num_inducing, kernel, epochs
        self.batch_size, self.lr, self.seed = batch_size, lr, seed
        self.kernel_kwargs = dict(kernel_kwargs or {})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seconds = 0.0

    def fit(self, x, y) -> SVGP:
        t0 = time.perf_counter()
        torch.manual_seed(self.seed)
        self.std = _Standardizer()
        xs, ys = self.std.fit(x, y, self.device)
        n = xs.shape[0]
        idx = torch.randperm(n)[: min(self.num_inducing, n)]
        kernel = _kernel(self.kernel, xs, **self.kernel_kwargs)
        self.model = _SVGPModel(xs[idx].clone(), kernel).to(self.device, xs.dtype)
        self.lik = gpytorch.likelihoods.GaussianLikelihood().to(self.device, xs.dtype)
        self.model.train(), self.lik.train()
        opt = torch.optim.Adam([*self.model.parameters(), *self.lik.parameters()], lr=self.lr)
        elbo = gpytorch.mlls.VariationalELBO(self.lik, self.model, num_data=n)
        steps = max(1, self.epochs * int(np.ceil(n / self.batch_size)))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=self.lr * 0.05)
        for _ in range(self.epochs):
            for b in torch.randperm(n, device=self.device).split(self.batch_size):
                opt.zero_grad()
                loss = -elbo(self.model(xs[b]), ys[b])
                loss.backward()
                opt.step()
                sched.step()
        self.seconds = time.perf_counter() - t0
        return self

    @torch.no_grad()
    def predict(self, x) -> Prediction:
        self.model.eval(), self.lik.eval()
        means, variances = [], []
        for chunk in torch.split(self.std.tx(x), 8192):
            f = self.model(chunk)
            means.append(f.mean)
            variances.append(f.variance)
        return self.std.out(torch.cat(means), torch.cat(variances), float(self.lik.noise))


class _DGPLayer(gpytorch.models.deep_gps.DeepGPLayer):
    def __init__(self, d_in: int, d_out: int | None, num_inducing: int, linear_mean: bool):
        shape = torch.Size([]) if d_out is None else torch.Size([d_out])
        inducing = torch.randn(*shape, num_inducing, d_in)
        q = gpytorch.variational.CholeskyVariationalDistribution(num_inducing, batch_shape=shape)
        s = gpytorch.variational.VariationalStrategy(self, inducing, q, learn_inducing_locations=True)
        super().__init__(s, d_in, d_out)
        self.mean_module = (
            gpytorch.means.LinearMean(d_in) if linear_mean else gpytorch.means.ConstantMean(batch_shape=shape)
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=shape, ard_num_dims=d_in), batch_shape=shape
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class _DGPModel(gpytorch.models.deep_gps.DeepGP):
    def __init__(self, d: int, hidden: int, num_inducing: int):
        super().__init__()
        self.hidden = _DGPLayer(d, hidden, num_inducing, True)
        self.out = _DGPLayer(hidden, None, num_inducing, False)

    def forward(self, x):
        return self.out(self.hidden(x))


class DeepGP:
    """Two-layer doubly stochastic deep GP (Salimbeni & Deisenroth, 2017)."""

    def __init__(
        self,
        hidden: int = 4,
        num_inducing: int = 128,
        epochs: int = 100,
        batch_size: int = 512,
        lr: float = 0.01,
        samples: int = 8,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        self.hidden, self.num_inducing, self.epochs = hidden, num_inducing, epochs
        self.batch_size, self.lr, self.samples, self.seed = batch_size, lr, samples, seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seconds = 0.0

    def fit(self, x, y) -> DeepGP:
        t0 = time.perf_counter()
        torch.manual_seed(self.seed)
        self.std = _Standardizer()
        xs, ys = self.std.fit(x, y, self.device)
        n = xs.shape[0]
        self.model = _DGPModel(xs.shape[1], self.hidden, self.num_inducing).to(self.device)
        self.lik = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model.train(), self.lik.train()
        opt = torch.optim.Adam([*self.model.parameters(), *self.lik.parameters()], lr=self.lr)
        elbo = gpytorch.mlls.DeepApproximateMLL(gpytorch.mlls.VariationalELBO(self.lik, self.model, num_data=n))
        with gpytorch.settings.num_likelihood_samples(self.samples):
            for _ in range(self.epochs):
                for b in torch.randperm(n, device=self.device).split(self.batch_size):
                    opt.zero_grad()
                    loss = -elbo(self.model(xs[b]), ys[b])
                    loss.backward()
                    opt.step()
        self.seconds = time.perf_counter() - t0
        return self

    @torch.no_grad()
    def predict(self, x) -> Prediction:
        self.model.eval(), self.lik.eval()
        means, variances = [], []
        with gpytorch.settings.num_likelihood_samples(32):
            for chunk in torch.split(self.std.tx(x), 4096):
                f = self.model(chunk)
                m, v = f.mean, f.variance
                means.append(m.mean(0))
                variances.append(v.mean(0) + m.var(0, unbiased=False))
        return self.std.out(torch.cat(means), torch.cat(variances), float(self.lik.noise))
