import torch

from mechanica import (
    constraint_jacobian,
    constraint_residual,
    project_positions,
    project_velocities,
    rattle_step,
)


def circle(q: torch.Tensor) -> torch.Tensor:
    return (q.square().sum() - 1).unsqueeze(0)


def test_constraint_projection_reaches_circle_and_tangent() -> None:
    q = project_positions(torch.tensor([1.2, 0.2], dtype=torch.float64), circle)
    v = project_velocities(q, torch.tensor([1.0, 2.0], dtype=torch.float64), circle)

    assert constraint_residual(circle, q).abs().max() < 1e-10
    assert (constraint_jacobian(circle, q) @ v).abs().max() < 1e-10


def test_rattle_preserves_position_and_velocity_constraints() -> None:
    q, v = rattle_step(
        lambda positions: torch.zeros_like(positions),
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        1.0,
        circle,
        0.1,
    )

    assert constraint_residual(circle, q).abs().max() < 1e-10
    assert (constraint_jacobian(circle, q) @ v).abs().max() < 1e-10
