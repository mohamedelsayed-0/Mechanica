"""Differentiable robotics kinematics utilities."""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def _as_link_lengths(link_lengths: Tensor, q: Tensor) -> Tensor:
    lengths = torch.as_tensor(link_lengths, dtype=q.dtype, device=q.device)
    if lengths.ndim != 1:
        raise ValueError("link_lengths must be a one-dimensional tensor")
    if lengths.shape[0] != q.shape[-1]:
        raise ValueError("link_lengths length must match q final dimension")
    return lengths


def _base_vector(base: Tensor | None, q: Tensor) -> Tensor:
    if base is None:
        return torch.zeros(2, dtype=q.dtype, device=q.device)
    base_tensor = torch.as_tensor(base, dtype=q.dtype, device=q.device)
    if base_tensor.shape != (2,):
        raise ValueError("base must have shape (2,)")
    return base_tensor


def planar_link_angles(q: Tensor) -> Tensor:
    """Return world-frame link angles for a planar serial chain."""
    if q.shape[-1] < 1:
        raise ValueError("q must contain at least one joint")
    return q.cumsum(dim=-1)


def planar_link_vectors(q: Tensor, link_lengths: Tensor) -> Tensor:
    """Return world-frame link displacement vectors shaped ``(..., links, 2)``."""
    lengths = _as_link_lengths(link_lengths, q)
    angles = planar_link_angles(q)
    vectors = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    return vectors * lengths.reshape(*(1 for _ in q.shape[:-1]), -1, 1)


def planar_forward_kinematics(
    q: Tensor,
    link_lengths: Tensor,
    *,
    base: Tensor | None = None,
) -> Tensor:
    """Return the end-effector position for a planar serial chain."""
    return planar_link_positions(q, link_lengths, base=base)[..., -1, :]


def planar_link_positions(
    q: Tensor,
    link_lengths: Tensor,
    *,
    base: Tensor | None = None,
) -> Tensor:
    """Return endpoint positions for every planar link."""
    vectors = planar_link_vectors(q, link_lengths)
    return vectors.cumsum(dim=-2) + _base_vector(base, q)


def planar_chain_points(
    q: Tensor,
    link_lengths: Tensor,
    *,
    base: Tensor | None = None,
) -> Tensor:
    """Return base point plus all link endpoint positions.

    The result has shape ``(..., links + 1, 2)`` and is useful for plotting or
    collision checks.
    """
    endpoints = planar_link_positions(q, link_lengths, base=base)
    base_point = _base_vector(base, q).expand(*q.shape[:-1], 2)
    return torch.cat([base_point.unsqueeze(-2), endpoints], dim=-2)


def planar_end_effector_pose(
    q: Tensor,
    link_lengths: Tensor,
    *,
    base: Tensor | None = None,
) -> Tensor:
    """Return planar end-effector pose ``[..., x, y, theta]``."""
    position = planar_forward_kinematics(q, link_lengths, base=base)
    theta = planar_link_angles(q)[..., -1:]
    return torch.cat([position, theta], dim=-1)


def planar_jacobian(
    q: Tensor,
    link_lengths: Tensor,
    *,
    link_index: int | None = None,
) -> Tensor:
    """Return translational Jacobian for a planar link endpoint.

    ``link_index`` is zero-based and defaults to the end effector. The returned
    tensor has shape ``(..., 2, joints)``.
    """
    lengths = _as_link_lengths(link_lengths, q)
    joints = q.shape[-1]
    if link_index is None:
        link_index = joints - 1
    if link_index < 0 or link_index >= joints:
        raise ValueError("link_index must select an existing link")

    angles = planar_link_angles(q)
    length_scale = lengths.reshape(*(1 for _ in q.shape[:-1]), -1, 1)
    pieces = length_scale * torch.stack([-torch.sin(angles), torch.cos(angles)], dim=-1)
    jacobian_columns = []
    for joint_index in range(joints):
        if joint_index > link_index:
            column = torch.zeros(*q.shape[:-1], 2, dtype=q.dtype, device=q.device)
        else:
            column = pieces[..., joint_index : link_index + 1, :].sum(dim=-2)
        jacobian_columns.append(column)

    return torch.stack(jacobian_columns, dim=-1)


def operational_space_velocity(jacobian: Tensor, qdot: Tensor) -> Tensor:
    """Return task-space velocity from a Jacobian and joint velocity."""
    if jacobian.shape[-1] != qdot.shape[-1]:
        raise ValueError("jacobian joint dimension must match qdot final dimension")
    return (jacobian @ qdot.unsqueeze(-1)).squeeze(-1)
