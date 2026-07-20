"""Differentiable primitive collision distances and native broad phase."""

from __future__ import annotations

import torch

from .classical import gravity_neighbor_list
from .robot_model import BOX, CYLINDER, MESH, RobotModel

Tensor = torch.Tensor


def sphere_signed_distance(
    first_center: Tensor,
    first_radius: float | Tensor,
    second_center: Tensor,
    second_radius: float | Tensor,
) -> Tensor:
    """Return positive separation and negative penetration between spheres."""
    radii = torch.as_tensor(first_radius, dtype=first_center.dtype, device=first_center.device)
    radii = radii + torch.as_tensor(second_radius, dtype=first_center.dtype, device=first_center.device)
    return torch.linalg.vector_norm(second_center - first_center, dim=-1) - radii


def point_segment_distance(point: Tensor, start: Tensor, end: Tensor) -> Tensor:
    """Return the Euclidean distance from points to finite segments."""
    segment = end - start
    denominator = segment.square().sum(-1).clamp_min(torch.finfo(point.dtype).eps)
    parameter = ((point - start) * segment).sum(-1).div(denominator).clamp(0, 1)
    closest = start + parameter.unsqueeze(-1) * segment
    return torch.linalg.vector_norm(point - closest, dim=-1)


def segment_segment_distance(
    first_start: Tensor,
    first_end: Tensor,
    second_start: Tensor,
    second_end: Tensor,
) -> Tensor:
    """Return the minimum distance between finite 3D segments."""
    first = first_end - first_start
    second = second_end - second_start
    offset = first_start - second_start
    a = first.square().sum(-1)
    b = (first * second).sum(-1)
    c = second.square().sum(-1)
    d = (first * offset).sum(-1)
    e = (second * offset).sum(-1)
    determinant = a * c - b.square()
    safe = determinant.clamp_min(torch.finfo(first.dtype).eps)
    s = ((b * e - c * d) / safe).clamp(0, 1)
    t = ((a * e - b * d) / safe).clamp(0, 1)
    interior = torch.linalg.vector_norm(offset + s.unsqueeze(-1) * first - t.unsqueeze(-1) * second, dim=-1)
    boundaries = torch.stack(
        (point_segment_distance(first_start, second_start, second_end),
         point_segment_distance(first_end, second_start, second_end),
         point_segment_distance(second_start, first_start, first_end),
         point_segment_distance(second_end, first_start, first_end)),
        -1,
    ).min(-1).values
    valid = (determinant > torch.finfo(first.dtype).eps) & (s > 0) & (s < 1) & (t > 0) & (t < 1)
    return torch.where(valid, torch.minimum(interior, boundaries), boundaries)


def capsule_signed_distance(
    first_start: Tensor,
    first_end: Tensor,
    first_radius: float | Tensor,
    second_start: Tensor,
    second_end: Tensor,
    second_radius: float | Tensor,
) -> Tensor:
    """Return signed separation between capsules."""
    radius = torch.as_tensor(first_radius, dtype=first_start.dtype, device=first_start.device)
    radius = radius + torch.as_tensor(second_radius, dtype=first_start.dtype, device=first_start.device)
    return segment_segment_distance(first_start, first_end, second_start, second_end) - radius


def plane_signed_distance(point: Tensor, normal: Tensor, offset: float | Tensor = 0.0) -> Tensor:
    """Return signed point distance to ``normal . point = offset``."""
    unit = normal / normal.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(point.dtype).eps)
    return (point * unit).sum(-1) - torch.as_tensor(offset, dtype=point.dtype, device=point.device)


def box_signed_distance(point: Tensor, pose: Tensor, half_extents: Tensor) -> Tensor:
    """Return the standard signed distance to an oriented box."""
    local = pose[..., :3, :3].transpose(-1, -2) @ (point - pose[..., :3, 3]).unsqueeze(-1)
    delta = local.squeeze(-1).abs() - half_extents.to(point)
    return torch.linalg.vector_norm(delta.clamp_min(0), dim=-1) + delta.max(-1).values.clamp_max(0)


def broad_phase_pairs(
    centers: Tensor,
    radii: Tensor,
    *,
    margin: float = 0.0,
    excluded_pairs: Tensor | None = None,
    use_native: bool | None = None,
) -> Tensor:
    """Return potentially overlapping sphere bounds using the native spatial hash."""
    radii = radii.to(centers)
    if centers.shape[0] == 0:
        return torch.empty((0, 2), dtype=torch.long, device=centers.device)
    cutoff = float((2 * radii.max() + margin).detach().cpu())
    pairs = gravity_neighbor_list(centers, cutoff, use_native=use_native)
    if pairs.numel() == 0:
        return pairs
    delta = centers[pairs[:, 1]] - centers[pairs[:, 0]]
    threshold = radii[pairs].sum(-1) + margin
    pairs = pairs[delta.square().sum(-1) <= threshold.square()]
    if excluded_pairs is not None and pairs.numel():
        excluded = {tuple(pair) for pair in excluded_pairs.detach().cpu().tolist()}
        keep = [tuple(pair) not in excluded for pair in pairs.detach().cpu().tolist()]
        pairs = pairs[torch.tensor(keep, dtype=torch.bool, device=pairs.device)]
    return pairs


def robot_collision_bounds(model: RobotModel, q: Tensor) -> tuple[Tensor, Tensor]:
    """Return conservative world-space sphere bounds for URDF collision geometry."""
    from .rigid_body import forward_kinematics

    if q.ndim != 1:
        raise ValueError("robot collision bounds currently expect an unbatched configuration")
    if model.collision_links.numel() == 0:
        return torch.empty(0, 3, dtype=q.dtype, device=q.device), torch.empty(0, dtype=q.dtype, device=q.device)
    poses = (
        forward_kinematics(model, q)[model.collision_links.to(q.device)]
        @ model.collision_poses.to(q)
    )
    parameters = model.collision_parameters.to(q)
    radii = parameters[:, 0]
    radii = torch.where(model.collision_types.to(q.device) == BOX, parameters.norm(dim=-1), radii)
    cylinder_radius = torch.sqrt(parameters[:, 0].square() + 0.25 * parameters[:, 1].square())
    radii = torch.where(model.collision_types.to(q.device) == CYLINDER, cylinder_radius, radii)
    radii = torch.where(model.collision_types.to(q.device) == MESH, parameters.norm(dim=-1), radii)
    return poses[:, :3, 3], radii


def robot_self_collision_pairs(
    model: RobotModel,
    q: Tensor,
    *,
    margin: float = 0.0,
    exclude_adjacent: bool = True,
    use_native: bool | None = None,
) -> Tensor:
    """Return broad-phase robot self-collision geometry pairs."""
    centers, radii = robot_collision_bounds(model, q)
    excluded = []
    for first in range(model.collision_links.numel()):
        for second in range(first + 1, model.collision_links.numel()):
            first_link = int(model.collision_links[first])
            second_link = int(model.collision_links[second])
            adjacent = int(model.parents[first_link]) == second_link or int(model.parents[second_link]) == first_link
            if first_link == second_link or (exclude_adjacent and adjacent):
                excluded.append((first, second))
    excluded_tensor = torch.tensor(excluded, dtype=torch.long, device=q.device).reshape(-1, 2)
    return broad_phase_pairs(
        centers, radii, margin=margin, excluded_pairs=excluded_tensor, use_native=use_native
    )


def smooth_collision_cost(distance: Tensor, *, margin: float = 0.0, softness: float = 0.01) -> Tensor:
    """Return a smooth nonnegative collision penalty."""
    return torch.nn.functional.softplus((margin - distance) / softness) * softness
