"""Fit an unknown spring constant from trajectory data."""

import torch

from mechanica import LagrangianSystem, fit_lagrangian_residual

m = torch.tensor(1.0)
true_k = torch.tensor(9.0)
learned_log_k = torch.nn.Parameter(torch.tensor(0.0))

t = torch.linspace(0, 3, 128)
w = torch.sqrt(true_k / m)
q = torch.cos(w * t).unsqueeze(-1)
qdot = -w * torch.sin(w * t).unsqueeze(-1)
qddot = -(w**2) * torch.cos(w * t).unsqueeze(-1)


def kinetic(q_sample: torch.Tensor, qdot_sample: torch.Tensor) -> torch.Tensor:
    return 0.5 * m * (qdot_sample * qdot_sample).sum()


def potential(q_sample: torch.Tensor) -> torch.Tensor:
    k = torch.exp(learned_log_k)
    return 0.5 * k * (q_sample * q_sample).sum()


system = LagrangianSystem(kinetic=kinetic, potential=potential)
result = fit_lagrangian_residual(
    system,
    q,
    qdot,
    qddot,
    [learned_log_k],
    steps=300,
    lr=1e-2,
)

print("final loss:", result.final_loss)
print("learned k:", torch.exp(learned_log_k).item())
