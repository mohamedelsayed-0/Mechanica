"""Trajectory diagnostics for mechanics models."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .analytical import LagrangianSystem

Tensor = torch.Tensor


@dataclass
class DiagnosticsReport:
    """A compact container for mechanics diagnostics."""

    metrics: dict[str, float] = field(default_factory=dict)
    tensors: dict[str, Tensor] = field(default_factory=dict)

    def add_metric(self, name: str, value: float | Tensor) -> None:
        if isinstance(value, Tensor):
            value = float(value.detach().cpu())
        self.metrics[name] = value

    def add_tensor(self, name: str, value: Tensor) -> None:
        self.tensors[name] = value

    def summary(self) -> str:
        return "\n".join(f"{name}: {value:.6g}" for name, value in self.metrics.items())


def residual_stats(residual: Tensor, *, prefix: str = "residual") -> dict[str, float]:
    """Return useful scalar stats for a residual tensor."""
    norms = residual.norm(dim=-1)
    return {
        f"{prefix}_mean": float(norms.mean().detach().cpu()),
        f"{prefix}_max": float(norms.max().detach().cpu()),
        f"{prefix}_rms": float(torch.sqrt((norms * norms).mean()).detach().cpu()),
    }


def energy_drift(energy: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Return normalized energy drift over a trajectory."""
    scale = energy.abs().mean().clamp_min(eps)
    return (energy.max() - energy.min()).abs() / scale


def lagrangian_diagnostics(
    system: LagrangianSystem,
    q: Tensor,
    qdot: Tensor,
    qddot: Tensor,
) -> DiagnosticsReport:
    """Evaluate a trajectory against a Lagrangian system."""
    residual = system.euler_lagrange_residual(q, qdot, qddot)
    energy = system.energy(q, qdot)

    report = DiagnosticsReport()
    report.add_tensor("euler_lagrange_residual", residual)
    report.add_tensor("energy", energy)
    for name, value in residual_stats(residual, prefix="euler_lagrange").items():
        report.add_metric(name, value)
    report.add_metric("energy_drift", energy_drift(energy))
    return report
