import pytest
import torch

from mechanica import HamiltonianSystem, LagrangianSystem, canonical_transformation_residual


def test_lagrangian_harmonic_oscillator_residual_is_small() -> None:
    m = torch.tensor(2.0)
    k = torch.tensor(8.0)

    def kinetic(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        return 0.5 * m * (qdot * qdot).sum()

    def potential(q: torch.Tensor) -> torch.Tensor:
        return 0.5 * k * (q * q).sum()

    system = LagrangianSystem(kinetic=kinetic, potential=potential)
    t = torch.linspace(0, 2, 32)
    w = torch.sqrt(k / m)
    q = torch.cos(w * t).unsqueeze(-1)
    qdot = -w * torch.sin(w * t).unsqueeze(-1)
    qddot = -(w**2) * torch.cos(w * t).unsqueeze(-1)

    residual = system.euler_lagrange_residual(q, qdot, qddot)

    assert residual.abs().max() < 1e-5


def test_lagrangian_generalized_momentum_matches_mass_velocity() -> None:
    m = torch.tensor(2.0)

    def kinetic(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        return 0.5 * m * (qdot * qdot).sum()

    system = LagrangianSystem(kinetic=kinetic)
    q = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    qdot = torch.tensor([[1.0, -1.0], [0.5, 2.0]])

    momentum = system.momentum(q, qdot)

    assert torch.allclose(momentum, m * qdot)


def test_noether_charge_and_symmetry_residual_for_rotation() -> None:
    m = torch.tensor(2.0)
    k = torch.tensor(3.0)

    def kinetic(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        return 0.5 * m * (qdot * qdot).sum()

    def potential(q: torch.Tensor) -> torch.Tensor:
        return 0.5 * k * (q * q).sum()

    def rotation(q: torch.Tensor) -> torch.Tensor:
        return torch.stack([-q[1], q[0]])

    system = LagrangianSystem(kinetic=kinetic, potential=potential)
    q = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    qdot = torch.tensor([[0.0, 3.0], [4.0, 0.0]])

    charge = system.noether_charge(q, qdot, rotation)
    residual = system.coordinate_symmetry_residual(q, qdot, rotation)

    expected_charge = m * (q[:, 0] * qdot[:, 1] - q[:, 1] * qdot[:, 0])
    assert torch.allclose(charge, expected_charge)
    assert residual.abs().max() < 1e-6


def test_hamiltonian_harmonic_oscillator_vector_field() -> None:
    m = torch.tensor(2.0)
    k = torch.tensor(8.0)

    def hamiltonian(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return 0.5 * (p * p).sum() / m + 0.5 * k * (q * q).sum()

    system = HamiltonianSystem(hamiltonian)
    q = torch.tensor([[0.5], [1.0]])
    p = torch.tensor([[2.0], [-1.0]])

    dqdt, dpdt = system.vector_field(q, p)

    assert torch.allclose(dqdt, p / m)
    assert torch.allclose(dpdt, -k * q)


def test_hamiltonian_poisson_bracket_and_time_derivative() -> None:
    m = torch.tensor(2.0)
    k = torch.tensor(8.0)

    def hamiltonian(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return 0.5 * (p * p).sum() / m + 0.5 * k * (q * q).sum()

    def position(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return q[0]

    def momentum(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return p[0]

    system = HamiltonianSystem(hamiltonian)
    q = torch.tensor([[0.5], [1.0]])
    p = torch.tensor([[2.0], [-1.0]])

    bracket = system.poisson_bracket(position, momentum, q, p)
    position_derivative = system.time_derivative(position, q, p)

    assert torch.allclose(bracket, torch.ones_like(bracket))
    assert torch.allclose(position_derivative, p.squeeze(-1) / m)


def test_canonical_transformation_residual_detects_phase_space_structure() -> None:
    q = torch.tensor([[1.0, 2.0]])
    p = torch.tensor([[3.0, 4.0]])

    def canonical_rotation(
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return p, -q

    def noncanonical_scaling(
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return 2 * q, p

    canonical = canonical_transformation_residual(canonical_rotation, q, p)
    noncanonical = canonical_transformation_residual(noncanonical_scaling, q, p)

    assert canonical.abs().max() < 1e-6
    assert noncanonical.abs().max() == pytest.approx(1.0)
