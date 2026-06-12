"""Small fitting helpers for inverse mechanics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from .analytical import LagrangianSystem

Tensor = torch.Tensor


@dataclass
class FitResult:
    """Optimization trace returned by fitting helpers."""

    losses: list[float]

    @property
    def final_loss(self) -> float:
        if not self.losses:
            return float("nan")
        return self.losses[-1]


def fit_lagrangian_residual(
    system: LagrangianSystem,
    q: Tensor,
    qdot: Tensor,
    qddot: Tensor,
    parameters: Iterable[torch.nn.Parameter | Tensor],
    *,
    steps: int = 500,
    lr: float = 1e-3,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
) -> FitResult:
    """Fit parameters by minimizing the Euler-Lagrange residual.

    Parameters are usually captured by the ``kinetic`` or ``potential`` callables
    used to build ``system``.
    """
    params = list(parameters)
    if not params:
        raise ValueError("at least one parameter is required")

    optimizer = optimizer_cls(params, lr=lr)
    losses: list[float] = []

    for _ in range(steps):
        optimizer.zero_grad()
        residual = system.euler_lagrange_residual(q, qdot, qddot, create_graph=True)
        loss = (residual * residual).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return FitResult(losses=losses)

