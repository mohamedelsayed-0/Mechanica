"""ML-facing wrappers and losses for learned mechanics models."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .analytical import HamiltonianSystem, LagrangianSystem, _scalar

Tensor = torch.Tensor
ScalarFn1 = Callable[[Tensor], Tensor]
ScalarFn2 = Callable[[Tensor, Tensor], Tensor]


class LagrangianModule(torch.nn.Module):
    """Wrap Torch callables as a trainable Lagrangian system."""

    def __init__(
        self,
        *,
        kinetic: ScalarFn2 | torch.nn.Module | None = None,
        potential: ScalarFn1 | torch.nn.Module | None = None,
        lagrangian: ScalarFn2 | torch.nn.Module | None = None,
        name: str = "learned_lagrangian",
    ) -> None:
        super().__init__()
        if lagrangian is None and kinetic is None:
            raise ValueError("provide either lagrangian or kinetic")
        self.kinetic = kinetic
        self.potential = potential
        self.lagrangian_model = lagrangian
        self.name = name

    def forward(self, q: Tensor, qdot: Tensor) -> Tensor:
        """Evaluate the scalar Lagrangian for one sample."""
        if self.lagrangian_model is not None:
            return _scalar(self.lagrangian_model(q, qdot), "lagrangian")

        kinetic = _scalar(self.kinetic(q, qdot), "kinetic")  # type: ignore[misc]
        potential = torch.zeros((), dtype=q.dtype, device=q.device)
        if self.potential is not None:
            potential = _scalar(self.potential(q), "potential")
        return kinetic - potential

    def system(self) -> LagrangianSystem:
        """Return a ``LagrangianSystem`` view backed by this module."""
        return LagrangianSystem(lagrangian_fn=self.forward, name=self.name)


class HamiltonianModule(torch.nn.Module):
    """Wrap a Torch callable as a trainable Hamiltonian system."""

    def __init__(
        self,
        hamiltonian: ScalarFn2 | torch.nn.Module,
        *,
        name: str = "learned_hamiltonian",
    ) -> None:
        super().__init__()
        self.hamiltonian_model = hamiltonian
        self.name = name

    def forward(self, q: Tensor, p: Tensor) -> Tensor:
        """Evaluate the scalar Hamiltonian for one sample."""
        return _scalar(self.hamiltonian_model(q, p), "hamiltonian")

    def system(self) -> HamiltonianSystem:
        """Return a ``HamiltonianSystem`` view backed by this module."""
        return HamiltonianSystem(self.forward, name=self.name)


def residual_loss(residual: Tensor, *, reduction: str = "mean") -> Tensor:
    """Return a scalar loss from a vector residual tensor."""
    squared = residual * residual
    if reduction == "mean":
        return squared.mean()
    if reduction == "sum":
        return squared.sum()
    if reduction == "none":
        return squared
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def lagrangian_residual_loss(
    system: LagrangianSystem,
    q: Tensor,
    qdot: Tensor,
    qddot: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Return Euler-Lagrange residual loss for trajectory samples."""
    residual = system.euler_lagrange_residual(q, qdot, qddot, create_graph=True)
    return residual_loss(residual, reduction=reduction)


def hamiltonian_residual_loss(
    system: HamiltonianSystem,
    q: Tensor,
    p: Tensor,
    qdot: Tensor,
    pdot: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Return Hamilton-equation residual loss for trajectory samples."""
    residual = system.hamilton_equations_residual(q, p, qdot, pdot, create_graph=True)
    return residual_loss(residual, reduction=reduction)
