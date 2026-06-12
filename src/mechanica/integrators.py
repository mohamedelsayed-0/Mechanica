"""Small differentiable integrators for mechanics experiments."""

from __future__ import annotations

from collections.abc import Callable

import torch

Tensor = torch.Tensor
State = Tensor | tuple[Tensor, ...]


def _tree_add(left: State, right: State) -> State:
    if isinstance(left, tuple):
        if not isinstance(right, tuple) or len(left) != len(right):
            raise ValueError("state structures must match")
        return tuple(a + b for a, b in zip(left, right))
    if isinstance(right, tuple):
        raise ValueError("state structures must match")
    return left + right


def _tree_mul(value: State, scalar: float | Tensor) -> State:
    if isinstance(value, tuple):
        return tuple(item * scalar for item in value)
    return value * scalar


def euler_step(
    dynamics: Callable[[Tensor, State], State],
    t: Tensor,
    state: State,
    dt: float | Tensor,
) -> State:
    """One explicit Euler step for tensor or tuple states."""
    return _tree_add(state, _tree_mul(dynamics(t, state), dt))


def rk4_step(
    dynamics: Callable[[Tensor, State], State],
    t: Tensor,
    state: State,
    dt: float | Tensor,
) -> State:
    """One fourth-order Runge-Kutta step for tensor or tuple states."""
    step = torch.as_tensor(dt, dtype=t.dtype, device=t.device)
    half = step / 2
    k1 = dynamics(t, state)
    k2 = dynamics(t + half, _tree_add(state, _tree_mul(k1, half)))
    k3 = dynamics(t + half, _tree_add(state, _tree_mul(k2, half)))
    k4 = dynamics(t + step, _tree_add(state, _tree_mul(k3, step)))

    if isinstance(state, tuple):
        return tuple(
            s + step * (a + 2 * b + 2 * c + d) / 6
            for s, a, b, c, d in zip(state, k1, k2, k3, k4)  # type: ignore[arg-type]
        )
    return state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6  # type: ignore[operator]


def _acceleration(
    force_fn: Callable[..., Tensor],
    positions: Tensor,
    velocities: Tensor,
    masses: float | Tensor,
) -> Tensor:
    try:
        forces = force_fn(positions, velocities)
    except TypeError:
        forces = force_fn(positions)
    mass = torch.as_tensor(masses, dtype=positions.dtype, device=positions.device)
    while mass.ndim < forces[..., 0].ndim:
        mass = mass.unsqueeze(0)
    return forces / mass.unsqueeze(-1)


def semi_implicit_euler_step(
    force_fn: Callable[..., Tensor],
    positions: Tensor,
    velocities: Tensor,
    masses: float | Tensor,
    dt: float | Tensor,
) -> tuple[Tensor, Tensor]:
    """One semi-implicit Euler step for particle systems."""
    step = torch.as_tensor(dt, dtype=positions.dtype, device=positions.device)
    acceleration = _acceleration(force_fn, positions, velocities, masses)
    next_velocities = velocities + step * acceleration
    next_positions = positions + step * next_velocities
    return next_positions, next_velocities


def velocity_verlet_step(
    force_fn: Callable[..., Tensor],
    positions: Tensor,
    velocities: Tensor,
    masses: float | Tensor,
    dt: float | Tensor,
) -> tuple[Tensor, Tensor]:
    """One velocity Verlet step for particle systems."""
    step = torch.as_tensor(dt, dtype=positions.dtype, device=positions.device)
    acceleration = _acceleration(force_fn, positions, velocities, masses)
    next_positions = positions + step * velocities + 0.5 * step * step * acceleration
    next_acceleration = _acceleration(force_fn, next_positions, velocities, masses)
    next_velocities = velocities + 0.5 * step * (acceleration + next_acceleration)
    return next_positions, next_velocities

