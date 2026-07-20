"""Differentiable rigid-body tree kinematics, dynamics, and control."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._native import NativeExtensionUnavailable, native_springs_requested
from .robot_model import PRISMATIC, RobotModel
from .spatial import force_cross, skew, so3_exp, so3_log, transform

Tensor = torch.Tensor


@dataclass(frozen=True)
class IKTarget:
    link: int | str
    position: Tensor | None = None
    rotation: Tensor | None = None
    weight: float = 1.0


def _joint_motion(model: RobotModel, link: int, q: Tensor) -> Tensor:
    index = int(model.q_indices[link])
    if index < 0:
        identity = torch.eye(4, dtype=q.dtype, device=q.device)
        return identity.expand(*q.shape[:-1], 4, 4)
    value = model.multipliers[link] * q[..., index] + model.offsets[link]
    axis = model.axes[link].expand(*q.shape[:-1], 3)
    if int(model.joint_types[link]) == PRISMATIC:
        identity = torch.eye(3, dtype=q.dtype, device=q.device).expand(*q.shape[:-1], 3, 3)
        return transform(identity, axis * value.unsqueeze(-1))
    return transform(so3_exp(axis * value.unsqueeze(-1)), torch.zeros_like(axis))


def forward_kinematics(
    model: RobotModel,
    q: Tensor,
    *,
    use_native: bool | None = None,
) -> Tensor:
    """Return world transforms shaped ``(..., links, 4, 4)``."""
    if q.shape[-1] != model.dof:
        raise ValueError(f"q must end in dimension {model.dof}")
    requested = native_springs_requested() if use_native is None else use_native
    if requested:
        from ._native import forward_kinematics_native

        try:
            return forward_kinematics_native(model, q)
        except NativeExtensionUnavailable:
            if use_native is True:
                raise
    origins = model.joint_origins.to(q)
    world: list[Tensor] = []
    for link in range(model.links):
        local = origins[link] @ _joint_motion(model, link, q)
        parent = int(model.parents[link])
        world.append(local if parent < 0 else world[parent] @ local)
    return torch.stack(world, -3)


def geometric_jacobian(model: RobotModel, q: Tensor, link: int | str) -> Tensor:
    """Return an angular-first world-frame Jacobian shaped ``(..., 6, dof)``."""
    link = model.link_index(link) if isinstance(link, str) else link
    world = forward_kinematics(model, q)
    point = world[..., link, :3, 3]
    columns = [torch.zeros(*q.shape[:-1], 6, dtype=q.dtype, device=q.device) for _ in range(model.dof)]
    current = link
    while current >= 0:
        coordinate = int(model.q_indices[current])
        if coordinate >= 0:
            parent = int(model.parents[current])
            parent_world = (
                torch.eye(4, dtype=q.dtype, device=q.device).expand(*q.shape[:-1], 4, 4)
                if parent < 0 else world[..., parent, :, :]
            )
            joint_world = parent_world @ model.joint_origins[current].to(q)
            axis = (joint_world[..., :3, :3] @ model.axes[current].to(q)).squeeze(-1) if model.axes[current].ndim > 1 else joint_world[..., :3, :3] @ model.axes[current].to(q)
            scale = model.multipliers[current].to(q)
            if int(model.joint_types[current]) == PRISMATIC:
                column = torch.cat((torch.zeros_like(axis), axis), -1)
            else:
                linear = torch.linalg.cross(axis, point - joint_world[..., :3, 3])
                column = torch.cat((axis, linear), -1)
            columns[coordinate] = columns[coordinate] + scale * column
        current = int(model.parents[current]) if current >= 0 else -1
    return torch.stack(columns, -1)


def _motion_transform(value: Tensor) -> Tensor:
    rotation = value[..., :3, :3].transpose(-1, -2)
    translation = value[..., :3, 3]
    zero = torch.zeros_like(rotation)
    return torch.cat(
        (torch.cat((rotation, zero), -1), torch.cat((-rotation @ skew(translation), rotation), -1)),
        -2,
    )


def spatial_inertias(model: RobotModel) -> Tensor:
    """Return angular-first spatial inertia matrices in link frames."""
    mass = model.masses
    cross = skew(model.centers_of_mass)
    upper_left = model.inertias + mass[:, None, None] * (cross @ cross.transpose(-1, -2))
    upper_right = mass[:, None, None] * cross
    lower_right = mass[:, None, None] * torch.eye(3, dtype=mass.dtype, device=mass.device)
    return torch.cat(
        (torch.cat((upper_left, upper_right), -1),
         torch.cat((upper_right.transpose(-1, -2), lower_right), -1)),
        -2,
    )


def _tree_terms(model: RobotModel, q: Tensor) -> tuple[list[Tensor], list[Tensor]]:
    local = []
    subspaces = []
    for link in range(model.links):
        value = model.joint_origins[link].to(q) @ _joint_motion(model, link, q)
        local.append(_motion_transform(value))
        axis = model.axes[link].to(q) * model.multipliers[link].to(q)
        subspaces.append(
            torch.cat((torch.zeros_like(axis), axis))
            if int(model.joint_types[link]) == PRISMATIC
            else torch.cat((axis, torch.zeros_like(axis)))
        )
    return local, subspaces


def inverse_dynamics_rnea(
    model: RobotModel,
    q: Tensor,
    qdot: Tensor,
    qddot: Tensor,
    *,
    gravity: Tensor | None = None,
    external_forces: Tensor | None = None,
) -> Tensor:
    """Compute generalized forces with the linear-time recursive Newton–Euler algorithm."""
    if q.shape != qdot.shape or q.shape != qddot.shape:
        raise ValueError("q, qdot, and qddot must have matching shapes")
    if q.ndim > 1:
        sample_shape = q.shape[:-1]
        flat_external = None if external_forces is None else external_forces.reshape(-1, model.links, 6)
        values = [
            inverse_dynamics_rnea(
                model,
                q_i,
                qdot_i,
                qddot_i,
                gravity=gravity,
                external_forces=None if flat_external is None else flat_external[index],
            )
            for index, (q_i, qdot_i, qddot_i) in enumerate(
                zip(q.reshape(-1, model.dof), qdot.reshape(-1, model.dof), qddot.reshape(-1, model.dof))
            )
        ]
        return torch.stack(values).reshape(*sample_shape, model.dof)
    gravity = torch.tensor([0, 0, -9.80665], dtype=q.dtype, device=q.device) if gravity is None else gravity.to(q)
    inertias = spatial_inertias(model).to(q)
    transforms, subspaces = _tree_terms(model, q)
    velocities: list[Tensor] = []
    accelerations: list[Tensor] = []
    forces: list[Tensor] = []
    base_acceleration = torch.cat((torch.zeros_like(gravity), -gravity))
    for link in range(model.links):
        parent = int(model.parents[link])
        coordinate = int(model.q_indices[link])
        joint_velocity = torch.zeros((), dtype=q.dtype, device=q.device) if coordinate < 0 else qdot[coordinate]
        joint_acceleration = torch.zeros_like(joint_velocity) if coordinate < 0 else qddot[coordinate]
        velocity = subspaces[link] * joint_velocity
        acceleration = subspaces[link] * joint_acceleration
        if parent < 0:
            acceleration = acceleration + base_acceleration
        else:
            velocity = transforms[link] @ velocities[parent] + velocity
            acceleration = (
                transforms[link] @ accelerations[parent]
                + acceleration
                + _motion_cross_vector(velocity, subspaces[link] * joint_velocity)
            )
        momentum = inertias[link] @ velocity
        force = inertias[link] @ acceleration + force_cross(velocity) @ momentum
        if external_forces is not None:
            force = force - external_forces[link].to(q)
        velocities.append(velocity)
        accelerations.append(acceleration)
        forces.append(force)

    generalized = [torch.zeros((), dtype=q.dtype, device=q.device) for _ in range(model.dof)]
    for link in range(model.links - 1, -1, -1):
        coordinate = int(model.q_indices[link])
        if coordinate >= 0:
            generalized[coordinate] = generalized[coordinate] + subspaces[link] @ forces[link]
        parent = int(model.parents[link])
        if parent >= 0:
            forces[parent] = forces[parent] + transforms[link].T @ forces[link]
    return torch.stack(generalized)


def _motion_cross_vector(left: Tensor, right: Tensor) -> Tensor:
    angular, linear = left.split(3)
    right_angular, right_linear = right.split(3)
    return torch.cat((torch.linalg.cross(angular, right_angular),
                      torch.linalg.cross(linear, right_angular) + torch.linalg.cross(angular, right_linear)))


def mass_matrix_crba(model: RobotModel, q: Tensor) -> Tensor:
    """Compute the joint-space mass matrix with the composite rigid-body algorithm."""
    if q.ndim > 1:
        values = [mass_matrix_crba(model, item) for item in q.reshape(-1, model.dof)]
        return torch.stack(values).reshape(*q.shape[:-1], model.dof, model.dof)
    transforms, subspaces = _tree_terms(model, q)
    composite = list(spatial_inertias(model).to(q).unbind(0))
    matrix = torch.zeros(model.dof, model.dof, dtype=q.dtype, device=q.device)
    for link in range(model.links - 1, -1, -1):
        coordinate = int(model.q_indices[link])
        if coordinate >= 0:
            force = composite[link] @ subspaces[link]
            matrix[coordinate, coordinate] += subspaces[link] @ force
            child = link
            ancestor = int(model.parents[child])
            while ancestor >= 0:
                force = transforms[child].T @ force
                other = int(model.q_indices[ancestor])
                if other >= 0:
                    value = subspaces[ancestor] @ force
                    if other == coordinate:
                        matrix[coordinate, coordinate] += 2 * value
                    else:
                        matrix[coordinate, other] += value
                        matrix[other, coordinate] += value
                child, ancestor = ancestor, int(model.parents[ancestor])
        parent = int(model.parents[link])
        if parent >= 0:
            composite[parent] = composite[parent] + transforms[link].T @ composite[link] @ transforms[link]
    return matrix


def forward_dynamics_aba(
    model: RobotModel,
    q: Tensor,
    qdot: Tensor,
    forces: Tensor,
    *,
    gravity: Tensor | None = None,
) -> Tensor:
    """Compute accelerations with ABA, falling back to CRBA for coupled mimic joints."""
    if q.shape != qdot.shape or q.shape != forces.shape:
        raise ValueError("q, qdot, and forces must have matching shapes")
    if q.ndim > 1:
        values = [
            forward_dynamics_aba(model, q_i, qdot_i, force_i, gravity=gravity)
            for q_i, qdot_i, force_i in zip(
                q.reshape(-1, model.dof),
                qdot.reshape(-1, model.dof),
                forces.reshape(-1, model.dof),
            )
        ]
        return torch.stack(values).reshape_as(q)
    active = model.q_indices[model.q_indices >= 0]
    if active.unique().numel() != active.numel():
        bias = inverse_dynamics_rnea(model, q, qdot, torch.zeros_like(q), gravity=gravity)
        return torch.linalg.solve(mass_matrix_crba(model, q), forces - bias)
    gravity = torch.tensor([0, 0, -9.80665], dtype=q.dtype, device=q.device) if gravity is None else gravity.to(q)
    transforms, subspaces = _tree_terms(model, q)
    articulated = list(spatial_inertias(model).to(q).unbind(0))
    velocities: list[Tensor] = []
    bias_accelerations: list[Tensor] = []
    bias_forces: list[Tensor] = []
    for link in range(model.links):
        parent = int(model.parents[link])
        coordinate = int(model.q_indices[link])
        speed = torch.zeros((), dtype=q.dtype, device=q.device) if coordinate < 0 else qdot[coordinate]
        joint_velocity = subspaces[link] * speed
        velocity = joint_velocity if parent < 0 else transforms[link] @ velocities[parent] + joint_velocity
        bias_acceleration = _motion_cross_vector(velocity, joint_velocity)
        bias_force = force_cross(velocity) @ (articulated[link] @ velocity)
        velocities.append(velocity)
        bias_accelerations.append(bias_acceleration)
        bias_forces.append(bias_force)

    u: list[Tensor | None] = [None] * model.links
    d: list[Tensor | None] = [None] * model.links
    capital_u: list[Tensor | None] = [None] * model.links
    for link in range(model.links - 1, -1, -1):
        parent = int(model.parents[link])
        coordinate = int(model.q_indices[link])
        reduced_inertia, reduced_bias = articulated[link], bias_forces[link]
        if coordinate >= 0:
            capital_u[link] = articulated[link] @ subspaces[link]
            d[link] = subspaces[link] @ capital_u[link]
            u[link] = forces[coordinate] - subspaces[link] @ bias_forces[link]
            reduced_inertia = articulated[link] - torch.outer(capital_u[link], capital_u[link]) / d[link]
            reduced_bias = bias_forces[link] + reduced_inertia @ bias_accelerations[link] + capital_u[link] * u[link] / d[link]
        if parent >= 0:
            articulated[parent] = articulated[parent] + transforms[link].T @ reduced_inertia @ transforms[link]
            bias_forces[parent] = bias_forces[parent] + transforms[link].T @ reduced_bias

    accelerations: list[Tensor] = []
    result = [torch.zeros((), dtype=q.dtype, device=q.device) for _ in range(model.dof)]
    base = torch.cat((torch.zeros_like(gravity), -gravity))
    for link in range(model.links):
        parent = int(model.parents[link])
        coordinate = int(model.q_indices[link])
        acceleration = (base if parent < 0 else transforms[link] @ accelerations[parent]) + bias_accelerations[link]
        if coordinate >= 0:
            result[coordinate] = (u[link] - capital_u[link] @ acceleration) / d[link]
            acceleration = acceleration + subspaces[link] * result[coordinate]
        accelerations.append(acceleration)
    return torch.stack(result)


def inverse_kinematics(
    model: RobotModel,
    initial_q: Tensor,
    link: int | str,
    target_position: Tensor,
    *,
    target_rotation: Tensor | None = None,
    damping: float = 1e-4,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    joint_limit_gain: float = 0.0,
) -> Tensor:
    """Solve position or pose IK with damped least squares and joint limits."""
    return inverse_kinematics_tasks(
        model,
        initial_q,
        [IKTarget(link, target_position, target_rotation)],
        damping=damping,
        max_iterations=max_iterations,
        tolerance=tolerance,
        joint_limit_gain=joint_limit_gain,
    )


def joint_limit_avoidance(q: Tensor, limits: Tensor) -> Tensor:
    """Return a bounded direction away from finite joint limits."""
    limits = limits.to(q)
    lower_distance = (q - limits[:, 0]).clamp_min(1e-4)
    upper_distance = (limits[:, 1] - q).clamp_min(1e-4)
    finite = torch.isfinite(limits).all(-1)
    direction = lower_distance.reciprocal().square() - upper_distance.reciprocal().square()
    return torch.where(finite, direction, torch.zeros_like(direction)).tanh()


def null_space_projector(jacobian: Tensor, pseudoinverse: Tensor | None = None) -> Tensor:
    """Return ``I - J# J`` for secondary IK and control tasks."""
    if pseudoinverse is None:
        pseudoinverse = torch.linalg.pinv(jacobian)
    identity = torch.eye(jacobian.shape[-1], dtype=jacobian.dtype, device=jacobian.device)
    return identity - pseudoinverse @ jacobian


def inverse_kinematics_tasks(
    model: RobotModel,
    initial_q: Tensor,
    targets: list[IKTarget],
    *,
    damping: float = 1e-4,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    joint_limit_gain: float = 0.0,
) -> Tensor:
    """Solve weighted multi-frame IK with an optional joint-limit null-space task."""
    if not targets:
        raise ValueError("at least one IK target is required")
    q = initial_q
    for _ in range(max_iterations):
        poses = forward_kinematics(model, q)
        errors, jacobians = [], []
        for target in targets:
            link_index = model.link_index(target.link) if isinstance(target.link, str) else target.link
            pose = poses[link_index]
            jacobian = geometric_jacobian(model, q, link_index)
            parts, rows = [], []
            if target.rotation is not None:
                parts.append(so3_log(target.rotation.to(q) @ pose[:3, :3].T))
                rows.append(jacobian[:3])
            if target.position is not None:
                parts.append(target.position.to(q) - pose[:3, 3])
                rows.append(jacobian[3:])
            if not parts:
                raise ValueError("each IK target needs a position or rotation")
            errors.append(target.weight * torch.cat(parts))
            jacobians.append(target.weight * torch.cat(rows))
        error = torch.cat(errors)
        jacobian = torch.cat(jacobians)
        if error.detach().norm().item() <= tolerance:
            break
        identity = torch.eye(error.numel(), dtype=q.dtype, device=q.device)
        pseudoinverse = jacobian.T @ torch.linalg.solve(
            jacobian @ jacobian.T + damping * identity, identity
        )
        change = pseudoinverse @ error
        if joint_limit_gain:
            change = change + joint_limit_gain * null_space_projector(jacobian, pseudoinverse) @ joint_limit_avoidance(q, model.limits)
        q = q + change
        finite = torch.isfinite(model.limits).all(-1).to(q.device)
        q = torch.where(finite, q.clamp(model.limits[:, 0].to(q), model.limits[:, 1].to(q)), q)
    return q


def operational_space_inertia(mass_matrix: Tensor, jacobian: Tensor, damping: float = 1e-8) -> Tensor:
    """Return Cartesian inertia ``(J M^-1 J.T)^-1``."""
    inverse_mass_jt = torch.linalg.solve(mass_matrix, jacobian.T)
    value = jacobian @ inverse_mass_jt
    return torch.linalg.inv(value + damping * torch.eye(value.shape[-1], dtype=value.dtype, device=value.device))


def dynamically_consistent_pseudoinverse(
    mass_matrix: Tensor, jacobian: Tensor, damping: float = 1e-8
) -> Tensor:
    """Return the dynamically consistent inverse ``M^-1 J.T Lambda``."""
    return torch.linalg.solve(mass_matrix, jacobian.T) @ operational_space_inertia(mass_matrix, jacobian, damping)


def cartesian_impedance(
    jacobian: Tensor,
    qdot: Tensor,
    position_error: Tensor,
    *,
    kp: float | Tensor,
    kd: float | Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Map a Cartesian PD wrench to generalized forces."""
    wrench = torch.as_tensor(kp, dtype=qdot.dtype, device=qdot.device) * position_error
    wrench = wrench - torch.as_tensor(kd, dtype=qdot.dtype, device=qdot.device) * (jacobian @ qdot)
    result = jacobian.T @ wrench
    return result if bias is None else result + bias
