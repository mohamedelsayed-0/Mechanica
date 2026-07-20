"""Batched Lie-group and spatial-vector operations for robotics."""

from __future__ import annotations

import torch

from ._native import NativeExtensionUnavailable, native_springs_requested

Tensor = torch.Tensor


def skew(vector: Tensor) -> Tensor:
    """Return the cross-product matrix of vectors shaped ``(..., 3)``."""
    if vector.shape[-1] != 3:
        raise ValueError("vector must end in dimension 3")
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape, 3)


def _coefficients(theta2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    small = theta2 < 1e-8
    safe = theta2.clamp_min(torch.finfo(theta2.dtype).tiny)
    theta = safe.sqrt()
    a = torch.where(small, 1 - theta2 / 6 + theta2.square() / 120, theta.sin() / theta)
    b = torch.where(small, 0.5 - theta2 / 24 + theta2.square() / 720, (1 - theta.cos()) / safe)
    c = torch.where(
        small,
        1 / 6 - theta2 / 120 + theta2.square() / 5040,
        (theta - theta.sin()) / (safe * theta),
    )
    return a, b, c


def so3_exp(rotation_vector: Tensor, *, use_native: bool | None = None) -> Tensor:
    """Map rotation vectors shaped ``(..., 3)`` to rotation matrices."""
    requested = native_springs_requested() if use_native is None else use_native
    if requested:
        from ._native import so3_exp_native

        try:
            return so3_exp_native(rotation_vector)
        except NativeExtensionUnavailable:
            if use_native is True:
                raise
    matrix = skew(rotation_vector)
    theta2 = rotation_vector.square().sum(-1, keepdim=True)
    a, b, _ = _coefficients(theta2)
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    return identity + a.unsqueeze(-1) * matrix + b.unsqueeze(-1) * (matrix @ matrix)


def so3_log(rotation: Tensor) -> Tensor:
    """Map rotation matrices shaped ``(..., 3, 3)`` to rotation vectors."""
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must be shaped (..., 3, 3)")
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    theta = cosine.acos()
    vector = torch.stack(
        (rotation[..., 2, 1] - rotation[..., 1, 2],
         rotation[..., 0, 2] - rotation[..., 2, 0],
         rotation[..., 1, 0] - rotation[..., 0, 1]),
        -1,
    )
    scale = torch.where(theta.abs() < 1e-6, 0.5 + theta.square() / 12, theta / (2 * theta.sin()))
    return scale.unsqueeze(-1) * vector


def so3_left_jacobian(rotation_vector: Tensor) -> Tensor:
    """Return the left Jacobian of ``SO(3)``."""
    matrix = skew(rotation_vector)
    theta2 = rotation_vector.square().sum(-1, keepdim=True)
    _, b, c = _coefficients(theta2)
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    return identity + b.unsqueeze(-1) * matrix + c.unsqueeze(-1) * (matrix @ matrix)


def transform(rotation: Tensor, translation: Tensor) -> Tensor:
    """Build homogeneous transforms from matching rotations and translations."""
    if rotation.shape[:-2] != translation.shape[:-1] or translation.shape[-1] != 3:
        raise ValueError("rotation and translation batch shapes must match")
    bottom = torch.zeros(*translation.shape[:-1], 1, 4, dtype=rotation.dtype, device=rotation.device)
    bottom[..., 0, 3] = 1
    return torch.cat((torch.cat((rotation, translation.unsqueeze(-1)), -1), bottom), -2)


def se3_exp(twist: Tensor) -> Tensor:
    """Map twists ``[..., translation, rotation]`` to homogeneous transforms."""
    if twist.shape[-1] != 6:
        raise ValueError("twist must end in dimension 6")
    translation, rotation_vector = twist.split(3, -1)
    return transform(so3_exp(rotation_vector), (so3_left_jacobian(rotation_vector) @ translation.unsqueeze(-1)).squeeze(-1))


def se3_log(value: Tensor) -> Tensor:
    """Map homogeneous transforms to twists ``[..., translation, rotation]``."""
    if value.shape[-2:] != (4, 4):
        raise ValueError("value must be shaped (..., 4, 4)")
    rotation_vector = so3_log(value[..., :3, :3])
    translation = torch.linalg.solve(so3_left_jacobian(rotation_vector), value[..., :3, 3])
    return torch.cat((translation, rotation_vector), -1)


def adjoint(value: Tensor) -> Tensor:
    """Return the ``SE(3)`` adjoint for angular-first spatial vectors."""
    rotation = value[..., :3, :3]
    translation = value[..., :3, 3]
    zero = torch.zeros_like(rotation)
    return torch.cat((torch.cat((rotation, zero), -1), torch.cat((skew(translation) @ rotation, rotation), -1)), -2)


def motion_cross(motion: Tensor) -> Tensor:
    """Return the spatial motion cross-product matrix."""
    angular, linear = motion.split(3, -1)
    zero = torch.zeros(*motion.shape[:-1], 3, 3, dtype=motion.dtype, device=motion.device)
    angular_cross = skew(angular)
    return torch.cat((torch.cat((angular_cross, zero), -1), torch.cat((skew(linear), angular_cross), -1)), -2)


def force_cross(motion: Tensor) -> Tensor:
    """Return the dual spatial force cross-product matrix."""
    return -motion_cross(motion).transpose(-1, -2)
