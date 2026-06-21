import pytest
import torch

import mechanica.classical as classical
from mechanica import (
    NativeExtensionUnavailable,
    center_of_mass,
    hooke_spring_force,
    kinetic_energy,
    linear_momentum,
    newton_residual,
    pairwise_gravity_force,
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


def test_pairwise_gravity_force_balances_pair() -> None:
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    masses = torch.tensor([2.0, 3.0])

    forces = pairwise_gravity_force(positions, masses, gravitational_constant=10.0)

    assert torch.allclose(forces[0], torch.tensor([15.0, 0.0]))
    assert torch.allclose(forces[1], torch.tensor([-15.0, 0.0]))
    assert torch.allclose(forces.sum(dim=0), torch.zeros(2))


def test_hooke_spring_force_explicit_native_failure_is_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_native(*args, **kwargs):
        raise NativeExtensionUnavailable("native test failure")

    monkeypatch.setattr(classical, "hooke_spring_force_native", fail_native)

    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    edges = torch.tensor([[0, 1]])

    with pytest.raises(NativeExtensionUnavailable, match="native test failure"):
        hooke_spring_force(positions, edges, rest_lengths=1.0, stiffness=10.0, use_native=True)


def test_hooke_spring_force_env_native_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_native(*args, **kwargs):
        raise NativeExtensionUnavailable("native test failure")

    monkeypatch.setenv("MECHANICA_USE_NATIVE", "1")
    monkeypatch.setattr(classical, "hooke_spring_force_native", fail_native)

    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    edges = torch.tensor([[0, 1]])

    with pytest.warns(RuntimeWarning, match="native test failure"):
        forces = hooke_spring_force(positions, edges, rest_lengths=1.0, stiffness=10.0)

    assert torch.allclose(forces[0], torch.tensor([10.0, 0.0]))
    assert torch.allclose(forces[1], torch.tensor([-10.0, 0.0]))


def test_pairwise_gravity_force_env_native_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_native(*args, **kwargs):
        raise NativeExtensionUnavailable("native gravity test failure")

    monkeypatch.setenv("MECHANICA_USE_NATIVE", "1")
    monkeypatch.setattr(classical, "pairwise_gravity_force_native", fail_native)

    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    masses = torch.tensor([2.0, 3.0])

    with pytest.warns(RuntimeWarning, match="native gravity test failure"):
        forces = pairwise_gravity_force(positions, masses, gravitational_constant=10.0)

    assert torch.allclose(forces[0], torch.tensor([15.0, 0.0]))
    assert torch.allclose(forces[1], torch.tensor([-15.0, 0.0]))


def test_newton_residual_zero_when_force_matches_acceleration() -> None:
    accelerations = torch.tensor([[0.0, -9.0], [1.0, 0.0]])
    masses = torch.tensor([2.0, 3.0])
    forces = accelerations * masses[:, None]

    residual = newton_residual(accelerations, forces, masses)

    assert torch.allclose(residual, torch.zeros_like(residual))
