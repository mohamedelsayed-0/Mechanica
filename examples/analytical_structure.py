"""Inspect conservation and phase-space structure with autograd."""

import torch

from mechanica import HamiltonianSystem, LagrangianSystem, canonical_transformation_residual

m = torch.tensor(1.0)
k = torch.tensor(4.0)


def kinetic(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
    return 0.5 * m * (qdot * qdot).sum()


def potential(q: torch.Tensor) -> torch.Tensor:
    return 0.5 * k * (q * q).sum()


def rotation(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([-q[1], q[0]])


lagrangian = LagrangianSystem(kinetic=kinetic, potential=potential)
q = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
qdot = torch.tensor([[0.0, 2.0], [-1.0, 0.0]])

angular_charge = lagrangian.noether_charge(q, qdot, rotation)
symmetry_error = lagrangian.coordinate_symmetry_residual(q, qdot, rotation)


def hamiltonian(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return 0.5 * (p * p).sum() / m + 0.5 * k * (q * q).sum()


def position(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return q[0]


def momentum(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return p[0]


hamiltonian_system = HamiltonianSystem(hamiltonian)
canonical_relation = hamiltonian_system.poisson_bracket(position, momentum, q[:, :1], qdot[:, :1])


def phase_rotation(q: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return p, -q


canonical_error = canonical_transformation_residual(phase_rotation, q, qdot)

print("angular charge", angular_charge)
print("rotation symmetry residual", symmetry_error)
print("position momentum poisson bracket", canonical_relation)
print("canonical transform residual max", canonical_error.abs().max())
