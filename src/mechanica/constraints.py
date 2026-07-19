"""Differentiable holonomic constraints and RATTLE integration."""

from __future__ import annotations

from collections.abc import Callable

import torch

Tensor = torch.Tensor
ConstraintFn = Callable[[Tensor], Tensor]
ForceFn = Callable[..., Tensor]


def constraint_residual(constraint_fn: ConstraintFn, positions: Tensor) -> Tensor:
    """Evaluate holonomic constraints as a flat residual vector."""
    value = constraint_fn(positions)
    if not isinstance(value, Tensor):
        raise TypeError("constraint_fn must return a tensor")
    return value.reshape(-1)


def constraint_jacobian(
    constraint_fn: ConstraintFn,
    positions: Tensor,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Return ``dc/dq`` with one row per scalar constraint."""
    jacobian = torch.autograd.functional.jacobian(
        lambda q: constraint_residual(constraint_fn, q),
        positions,
        create_graph=create_graph,
    )
    return jacobian.reshape(-1, positions.numel())


def _inverse_mass(masses: float | Tensor, positions: Tensor) -> Tensor:
    mass = torch.as_tensor(masses, dtype=positions.dtype, device=positions.device)
    if mass.ndim == 0:
        mass = mass.expand_as(positions)
    elif mass.shape == positions.shape[:-1]:
        mass = mass.unsqueeze(-1).expand_as(positions)
    else:
        mass = torch.broadcast_to(mass, positions.shape)
    if torch.any(mass <= 0):
        raise ValueError("masses must be positive")
    return mass.reshape(-1).reciprocal()


def project_positions(
    positions: Tensor,
    constraint_fn: ConstraintFn,
    masses: float | Tensor = 1.0,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 10,
    create_graph: bool = False,
) -> Tensor:
    """Project positions onto ``constraint_fn(q) = 0`` with Newton updates."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    inverse_mass = _inverse_mass(masses, positions)
    projected = positions
    for _ in range(max_iterations):
        residual = constraint_residual(constraint_fn, projected)
        if residual.numel() == 0 or residual.detach().abs().max().item() <= tolerance:
            break
        jacobian = constraint_jacobian(constraint_fn, projected, create_graph=create_graph)
        system = (jacobian * inverse_mass) @ jacobian.T
        multiplier = torch.linalg.solve(system, -residual)
        correction = inverse_mass * (jacobian.T @ multiplier)
        projected = projected + correction.reshape_as(projected)
    final_residual = constraint_residual(constraint_fn, projected)
    if final_residual.numel() and final_residual.detach().abs().max().item() > tolerance:
        raise RuntimeError("position projection did not converge")
    return projected


def project_velocities(
    positions: Tensor,
    velocities: Tensor,
    constraint_fn: ConstraintFn,
    masses: float | Tensor = 1.0,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Project velocities onto the tangent space ``dc/dq @ qdot = 0``."""
    if velocities.shape != positions.shape:
        raise ValueError("positions and velocities must have the same shape")
    inverse_mass = _inverse_mass(masses, positions)
    jacobian = constraint_jacobian(constraint_fn, positions, create_graph=create_graph)
    system = (jacobian * inverse_mass) @ jacobian.T
    violation = jacobian @ velocities.reshape(-1)
    multiplier = torch.linalg.solve(system, -violation)
    correction = inverse_mass * (jacobian.T @ multiplier)
    return velocities + correction.reshape_as(velocities)


def rattle_step(
    force_fn: ForceFn,
    positions: Tensor,
    velocities: Tensor,
    masses: float | Tensor,
    constraint_fn: ConstraintFn,
    dt: float | Tensor,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 10,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor]:
    """Advance a constrained particle system with a RATTLE-style Verlet step."""
    step = torch.as_tensor(dt, dtype=positions.dtype, device=positions.device)
    inverse_mass = _inverse_mass(masses, positions).reshape_as(positions)

    def acceleration(q: Tensor, v: Tensor) -> Tensor:
        try:
            force = force_fn(q, v)
        except TypeError:
            force = force_fn(q)
        if force.shape != q.shape:
            raise ValueError("force_fn must return the same shape as positions")
        return force * inverse_mass

    half_velocity = velocities + 0.5 * step * acceleration(positions, velocities)
    next_positions = project_positions(
        positions + step * half_velocity,
        constraint_fn,
        masses,
        tolerance=tolerance,
        max_iterations=max_iterations,
        create_graph=create_graph,
    )
    next_velocity = half_velocity + 0.5 * step * acceleration(next_positions, half_velocity)
    next_velocity = project_velocities(
        next_positions,
        next_velocity,
        constraint_fn,
        masses,
        create_graph=create_graph,
    )
    return next_positions, next_velocity
