"""Audit a simple harmonic oscillator with a Lagrangian residual."""

import torch

from mechanica import LagrangianSystem, lagrangian_diagnostics

m = torch.tensor(1.0)
k = torch.tensor(4.0)


def kinetic(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
    return 0.5 * m * (qdot * qdot).sum()


def potential(q: torch.Tensor) -> torch.Tensor:
    return 0.5 * k * (q * q).sum()


system = LagrangianSystem(kinetic=kinetic, potential=potential, name="harmonic_oscillator")

t = torch.linspace(0, 4, 128)
w = torch.sqrt(k / m)
q = torch.cos(w * t).unsqueeze(-1)
qdot = -w * torch.sin(w * t).unsqueeze(-1)
qddot = -(w**2) * torch.cos(w * t).unsqueeze(-1)

report = lagrangian_diagnostics(system, q, qdot, qddot)
print(report.summary())
