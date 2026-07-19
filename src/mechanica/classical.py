"""Classical mechanics helpers for particles and trajectories."""

from __future__ import annotations

import warnings

import torch

from ._native import (
    NativeExtensionUnavailable,
    hooke_spring_force_native,
    native_springs_requested,
    pairwise_gravity_force_native,
)

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


def pairwise_gravity_force(
    positions: Tensor,
    masses: float | Tensor,
    *,
    gravitational_constant: float = 1.0,
    softening: float = 0.0,
    cutoff: float | None = None,
    edges: Tensor | None = None,
    use_native: bool | None = None,
) -> Tensor:
    """Return pairwise inverse-square gravitational forces.

    ``positions`` is shaped ``(bodies, dim)``. Positive ``softening`` avoids
    singularities. ``cutoff`` builds a memory-efficient neighbor list; a
    reusable ``edges`` tensor can be supplied directly instead.
    """
    if cutoff is not None and edges is not None:
        raise ValueError("provide cutoff or edges, not both")
    if positions.ndim != 2:
        raise ValueError("positions must be shaped (bodies, dim)")

    fallback_on_unavailable = False
    if use_native is None:
        use_native = native_springs_requested()
        fallback_on_unavailable = use_native

    if use_native and cutoff is None and edges is None:
        try:
            return pairwise_gravity_force_native(
                positions,
                masses,
                gravitational_constant=gravitational_constant,
                softening=softening,
            )
        except NativeExtensionUnavailable as exc:
            if not fallback_on_unavailable:
                raise
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)

    mass = _as_tensor_like(masses, positions)
    if mass.ndim == 0:
        mass = mass.expand(positions.shape[0])
    if mass.ndim != 1 or mass.shape[0] != positions.shape[0]:
        raise ValueError("masses must be a scalar or shaped (bodies,)")

    if cutoff is not None:
        edges = gravity_neighbor_list(positions, cutoff, use_native=use_native)
    if edges is not None:
        edge_index = edges.to(dtype=torch.long, device=positions.device)
        if edge_index.ndim != 2 or edge_index.shape[-1] != 2:
            raise ValueError("edges must be shaped (pairs, 2)")
        i, j = edge_index.unbind(-1)
        delta = positions[j] - positions[i]
        distance2 = (delta * delta).sum(-1) + softening * softening
        inv_distance3 = distance2.clamp_min(torch.finfo(positions.dtype).tiny).pow(-1.5)
        force = (
            gravitational_constant
            * mass[i]
            * mass[j]
            * inv_distance3
        ).unsqueeze(-1) * delta
        forces = torch.zeros_like(positions)
        forces.index_add_(0, i, force)
        forces.index_add_(0, j, -force)
        return forces

    delta = positions.unsqueeze(0) - positions.unsqueeze(1)
    distance2 = (delta * delta).sum(dim=-1) + softening * softening
    eye = torch.eye(positions.shape[0], dtype=torch.bool, device=positions.device)
    distance2 = distance2.masked_fill(eye, 1)
    inv_distance3 = distance2.rsqrt() / distance2
    inv_distance3 = inv_distance3.masked_fill(eye, 0)

    mass_product = mass.unsqueeze(0) * mass.unsqueeze(1)
    magnitude = gravitational_constant * mass_product * inv_distance3
    return (magnitude.unsqueeze(-1) * delta).sum(dim=1)


def hooke_spring_force(
    positions: Tensor,
    edges: Tensor,
    rest_lengths: float | Tensor,
    stiffness: float | Tensor,
    *,
    velocities: Tensor | None = None,
    damping: float | Tensor = 0.0,
    use_native: bool | None = None,
) -> Tensor:
    """Return forces from Hooke springs over a particle graph.

    ``positions`` is shaped ``(..., bodies, dim)`` and ``edges`` is shaped
    ``(springs, 2)`` with integer particle indices. Edge parameters broadcast
    to ``(..., springs)``.

    Set ``use_native=True`` to route through the optional C++/Torch extension.
    When ``use_native`` is left as ``None``, setting ``MECHANICA_USE_NATIVE=1``
    enables the native path with automatic fallback to the pure Torch kernel if
    a compiler is unavailable.
    """
    fallback_on_unavailable = False
    if use_native is None:
        use_native = native_springs_requested()
        fallback_on_unavailable = use_native

    if use_native:
        try:
            return hooke_spring_force_native(
                positions,
                edges,
                rest_lengths,
                stiffness,
                velocities=velocities,
                damping=damping,
            )
        except NativeExtensionUnavailable as exc:
            if not fallback_on_unavailable:
                raise
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)

    if positions.ndim < 2:
        raise ValueError("positions must be shaped (..., bodies, dim)")
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError("edges must be shaped (springs, 2)")

    edge_index = edges.to(dtype=torch.long, device=positions.device)
    i, j = edge_index.unbind(-1)
    delta = positions[..., j, :] - positions[..., i, :]
    length = delta.norm(dim=-1).clamp_min(torch.finfo(positions.dtype).eps)
    direction = delta / length.unsqueeze(-1)

    rest = _as_tensor_like(rest_lengths, positions)
    k = _as_tensor_like(stiffness, positions)
    rest = torch.broadcast_to(rest, length.shape)
    k = torch.broadcast_to(k, length.shape)

    magnitude = k * (length - rest)

    if velocities is not None:
        if velocities.shape != positions.shape:
            raise ValueError("velocities must have the same shape as positions")
        rel_vel = velocities[..., j, :] - velocities[..., i, :]
        c = _as_tensor_like(damping, positions)
        magnitude = magnitude + torch.broadcast_to(c, length.shape) * (rel_vel * direction).sum(-1)

    force_edges = magnitude.unsqueeze(-1) * direction
    bodies, dim = positions.shape[-2:]
    batches = positions.numel() // (bodies * dim)
    offsets = torch.arange(batches, device=positions.device).unsqueeze(-1) * bodies
    flat_i = (i.unsqueeze(0) + offsets).reshape(-1)
    flat_j = (j.unsqueeze(0) + offsets).reshape(-1)
    flat_force = force_edges.reshape(-1, dim)
    forces = torch.zeros_like(positions).reshape(-1, dim)
    forces.index_add_(0, flat_i, flat_force)
    forces.index_add_(0, flat_j, -flat_force)
    return forces.reshape_as(positions)


def gravity_neighbor_list(
    positions: Tensor,
    cutoff: float,
    *,
    block_size: int = 1024,
    use_native: bool | None = None,
) -> Tensor:
    """Return unique body pairs separated by less than ``cutoff``.

    The fallback uses bounded ``block_size * bodies`` memory instead of a full
    pairwise displacement tensor.
    """
    if positions.ndim != 2:
        raise ValueError("positions must be shaped (bodies, dim)")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    requested = native_springs_requested() if use_native is None else use_native
    if requested and positions.device.type == "cpu":
        from ._native import gravity_neighbor_list_native

        try:
            return gravity_neighbor_list_native(positions, cutoff)
        except NativeExtensionUnavailable:
            if use_native is True:
                raise

    pairs = []
    count = positions.shape[0]
    indices = torch.arange(count, device=positions.device)
    cutoff2 = cutoff * cutoff
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        delta = positions[start:stop, None] - positions[None]
        distance2 = (delta * delta).sum(-1)
        mask = (distance2 < cutoff2) & (indices[None] > indices[start:stop, None])
        left, right = mask.nonzero(as_tuple=True)
        if left.numel():
            pairs.append(torch.stack((left + start, right), dim=-1))
    if pairs:
        return torch.cat(pairs)
    return torch.empty((0, 2), dtype=torch.long, device=positions.device)


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
