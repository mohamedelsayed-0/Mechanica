"""Kinematic estimates from sampled trajectories."""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def estimate_velocity(positions: Tensor, dt: float | Tensor) -> Tensor:
    """Estimate velocity using finite differences along the first axis."""
    if positions.shape[0] < 2:
        raise ValueError("at least two samples are required")
    step = torch.as_tensor(dt, dtype=positions.dtype, device=positions.device)
    velocity = torch.empty_like(positions)
    velocity[1:-1] = (positions[2:] - positions[:-2]) / (2 * step)
    velocity[0] = (positions[1] - positions[0]) / step
    velocity[-1] = (positions[-1] - positions[-2]) / step
    return velocity


def estimate_acceleration(positions: Tensor, dt: float | Tensor) -> Tensor:
    """Estimate acceleration using finite differences along the first axis."""
    return estimate_velocity(estimate_velocity(positions, dt), dt)

