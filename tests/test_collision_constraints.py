import torch

from mechanica.collision import (
    box_signed_distance,
    broad_phase_pairs,
    capsule_signed_distance,
    smooth_collision_cost,
    sphere_signed_distance,
)
from mechanica.constraints import constrained_forward_dynamics


def test_primitive_signed_distances_and_costs() -> None:
    first = torch.tensor([0.0, 0.0, 0.0])
    second = torch.tensor([3.0, 0.0, 0.0])
    assert torch.allclose(sphere_signed_distance(first, 1.0, second, 1.0), torch.tensor(1.0))

    capsule = capsule_signed_distance(first, torch.tensor([1.0, 0.0, 0.0]), 0.25,
                                       second, torch.tensor([4.0, 0.0, 0.0]), 0.25)
    assert torch.allclose(capsule, torch.tensor(1.5))
    assert smooth_collision_cost(torch.tensor(-0.1)) > smooth_collision_cost(torch.tensor(0.1))

    pose = torch.eye(4)
    assert box_signed_distance(torch.tensor([2.0, 0.0, 0.0]), pose, torch.ones(3)) == 1


def test_broad_phase_filters_distant_bounds() -> None:
    centers = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 0.0, 0.0]])
    pairs = broad_phase_pairs(centers, torch.ones(3), use_native=False)
    assert torch.equal(pairs, torch.tensor([[0, 1]]))


def test_constrained_dynamics_enforces_acceleration_constraint() -> None:
    acceleration, multiplier = constrained_forward_dynamics(
        torch.eye(2), torch.zeros(2), torch.tensor([1.0, 2.0]), torch.tensor([[1.0, 0.0]])
    )
    assert torch.allclose(acceleration, torch.tensor([0.0, 2.0]))
    assert torch.allclose(multiplier, torch.tensor([-1.0]))
