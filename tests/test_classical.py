import torch

from mechanica import (
    center_of_mass,
    hooke_spring_force,
    kinetic_energy,
    linear_momentum,
    newton_residual,
)


def test_kinetic_energy_and_momentum() -> None:
    velocities = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    masses = torch.tensor([2.0, 3.0])

    assert torch.allclose(kinetic_energy(velocities, masses), torch.tensor([1.0, 6.0]))
    assert torch.allclose(linear_momentum(velocities, masses), velocities * masses[:, None])


def test_center_of_mass() -> None:
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    masses = torch.tensor([1.0, 3.0])

    assert torch.allclose(center_of_mass(positions, masses), torch.tensor([1.5, 0.0]))


def test_hooke_spring_force_balances_pair() -> None:
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    edges = torch.tensor([[0, 1]])

    forces = hooke_spring_force(positions, edges, rest_lengths=1.0, stiffness=10.0)

    assert torch.allclose(forces[0], torch.tensor([10.0, 0.0]))
    assert torch.allclose(forces[1], torch.tensor([-10.0, 0.0]))


def test_newton_residual_zero_when_force_matches_acceleration() -> None:
    accelerations = torch.tensor([[0.0, -9.0], [1.0, 0.0]])
    masses = torch.tensor([2.0, 3.0])
    forces = accelerations * masses[:, None]

    residual = newton_residual(accelerations, forces, masses)

    assert torch.allclose(residual, torch.zeros_like(residual))
