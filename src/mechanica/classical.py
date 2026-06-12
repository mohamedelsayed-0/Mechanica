"""Classical mechanics helpers for particles and trajectories."""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def _as_tensor_like(value: float | Tensor, like: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _broadcast_mass(masses: float | Tensor, values_without_vector_dim: Tensor) -> Tensor:
    mass = _as_tensor_like(masses, values_without_vector_dim)
    while mass.ndim < values_without_vector_dim.ndim:
        mass = mass.unsqueeze(0)
    return mass


def kinetic_energy(velocities: Tensor, masses: float | Tensor = 1.0) -> Tensor:
    """Return kinetic energy for one body or a batch of body sets.

    ``velocities`` can be ``(..., dim)`` for one body or ``(..., bodies, dim)``
    for many bodies. For many bodies, energy is summed across the body axis.
    """
    speed2 = (velocities * velocities).sum(dim=-1)
    mass = _broadcast_mass(masses, speed2)
    energy = 0.5 * mass * speed2
    if energy.ndim == 0:
        return energy
    if velocities.ndim >= 3:
        return energy.sum(dim=-1)
    return energy


def linear_momentum(velocities: Tensor, masses: float | Tensor = 1.0) -> Tensor:
    """Return total linear momentum."""
    if velocities.ndim == 1:
        return _as_tensor_like(masses, velocities) * velocities

    speed_shape = velocities[..., 0]
    mass = _broadcast_mass(masses, speed_shape).unsqueeze(-1)
    momenta = mass * velocities
    if velocities.ndim >= 3:
        return momenta.sum(dim=-2)
    return momenta


def center_of_mass(positions: Tensor, masses: float | Tensor = 1.0) -> Tensor:
    """Return center of mass for positions shaped ``(..., bodies, dim)``."""
    if positions.ndim < 2:
        raise ValueError("positions must include at least body and coordinate dimensions")

    body_shape = positions[..., 0]
    mass = _broadcast_mass(masses, body_shape)
    weighted = positions * mass.unsqueeze(-1)
    total_mass = mass.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(positions.dtype).eps)
    return weighted.sum(dim=-2) / total_mass


def angular_momentum(
    positions: Tensor,
    velocities: Tensor,
    masses: float | Tensor = 1.0,
    *,
    origin: Tensor | None = None,
) -> Tensor:
    """Return angular momentum around ``origin``.

    In 2D this returns the scalar z-component. In 3D this returns a vector.
    Inputs are shaped ``(..., bodies, dim)``.
    """
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have the same shape")
    if positions.shape[-1] not in (2, 3):
        raise ValueError("angular_momentum supports 2D or 3D coordinates")

    if origin is None:
        origin = torch.zeros(positions.shape[-1], dtype=positions.dtype, device=positions.device)
    r = positions - origin
    p = linear_momentum(velocities, masses)

    if positions.ndim == 2:
        p = _as_tensor_like(masses, velocities[..., 0]).unsqueeze(-1) * velocities

    if positions.shape[-1] == 2:
        per_body = r[..., 0] * p[..., 1] - r[..., 1] * p[..., 0]
        return per_body.sum(dim=-1)

    return torch.cross(r, p, dim=-1).sum(dim=-2)


def near_surface_gravity_force(
    masses: float | Tensor,
    *,
    dim: int = 3,
    g: float = 9.80665,
    axis: int = 1,
    like: Tensor | None = None,
) -> Tensor:
    """Return a near-surface gravity force vector ``F = m g``.

    The gravity direction is negative along ``axis``.
    """
    if like is None:
        mass = torch.as_tensor(masses)
    else:
        mass = _as_tensor_like(masses, like)
    force = torch.zeros(*mass.shape, dim, dtype=mass.dtype, device=mass.device)
    force[..., axis] = -mass * g
    return force


def hooke_spring_force(
    positions: Tensor,
    edges: Tensor,
    rest_lengths: float | Tensor,
    stiffness: float | Tensor,
    *,
    velocities: Tensor | None = None,
    damping: float | Tensor = 0.0,
) -> Tensor:
    """Return forces from Hooke springs over a particle graph.

    ``positions`` is shaped ``(bodies, dim)`` and ``edges`` is shaped
    ``(springs, 2)`` with integer particle indices.
    """
    if positions.ndim != 2:
        raise ValueError("hooke_spring_force currently expects positions shaped (bodies, dim)")
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError("edges must be shaped (springs, 2)")

    i = edges[:, 0].long()
    j = edges[:, 1].long()
    delta = positions[j] - positions[i]
    length = delta.norm(dim=-1).clamp_min(torch.finfo(positions.dtype).eps)
    direction = delta / length.unsqueeze(-1)

    rest = _as_tensor_like(rest_lengths, positions)
    k = _as_tensor_like(stiffness, positions)
    while rest.ndim < length.ndim:
        rest = rest.unsqueeze(0)
    while k.ndim < length.ndim:
        k = k.unsqueeze(0)

    magnitude = k * (length - rest)

    if velocities is not None:
        rel_vel = velocities[j] - velocities[i]
        c = _as_tensor_like(damping, positions)
        magnitude = magnitude + c * (rel_vel * direction).sum(dim=-1)

    force_edges = magnitude.unsqueeze(-1) * direction
    forces = torch.zeros_like(positions)
    forces.index_add_(0, i, force_edges)
    forces.index_add_(0, j, -force_edges)
    return forces


def newton_residual(
    accelerations: Tensor,
    forces: Tensor,
    masses: float | Tensor = 1.0,
) -> Tensor:
    """Return ``m a - F`` for Newton's second law."""
    if accelerations.shape != forces.shape:
        raise ValueError("accelerations and forces must have the same shape")
    mass = _broadcast_mass(masses, accelerations[..., 0]).unsqueeze(-1)
    return mass * accelerations - forces
