"""Robotics dynamics helpers derived from Lagrangian mechanics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .analytical import LagrangianSystem, _flatten_state, _safe_grad
from .dynamics import join_state, split_state

Tensor = torch.Tensor


@dataclass
class ManipulatorTerms:
    """Manipulator equation terms ``M qddot + bias = tau``."""

    mass_matrix: Tensor
    bias_forces: Tensor


def _zero_velocity_like(q: Tensor) -> Tensor:
    return torch.zeros_like(q)


def _map_control_to_forces(
    control: Tensor,
    input_matrix: Tensor | None,
    reference_forces: Tensor,
) -> Tensor:
    control = control.to(dtype=reference_forces.dtype, device=reference_forces.device)
    if input_matrix is None:
        if control.shape != reference_forces.shape:
            raise ValueError(
                f"control must have shape {tuple(reference_forces.shape)}, "
                f"got {tuple(control.shape)}"
            )
        return control

    matrix = input_matrix.to(dtype=reference_forces.dtype, device=reference_forces.device)
    forces = matrix @ control if control.ndim == 1 else control @ matrix.T
    if forces.shape != reference_forces.shape:
        raise ValueError(
            f"mapped control forces must have shape {tuple(reference_forces.shape)}, "
            f"got {tuple(forces.shape)}"
        )
    return forces


def manipulator_terms(
    system: LagrangianSystem,
    q: Tensor,
    qdot: Tensor,
    *,
    create_graph: bool = False,
) -> ManipulatorTerms:
    """Return mass matrix and bias forces induced by a Lagrangian.

    The bias term contains velocity-coupling and potential-force terms, so
    inverse dynamics is ``tau = M @ qddot + bias``.
    """
    (q_flat, qdot_flat), sample_shape = _flatten_state(q, qdot)
    mass_matrices = []
    biases = []

    for q_i, v_i in zip(q_flat, qdot_flat):
        q_var = q_i.clone().requires_grad_(True)
        v_var = v_i.clone().requires_grad_(True)
        lagrangian = system.lagrangian(q_var, v_var)
        grad_q, grad_v = _safe_grad(
            lagrangian,
            (q_var, v_var),
            create_graph=True,
        )

        mixed_rows = []
        mass_rows = []
        for component in grad_v:
            h_vq_i, h_vv_i = _safe_grad(
                component,
                (q_var, v_var),
                create_graph=create_graph,
            )
            mixed_rows.append(h_vq_i)
            mass_rows.append(h_vv_i)

        mixed = torch.stack(mixed_rows, dim=0)
        mass_matrix = torch.stack(mass_rows, dim=0)
        bias = mixed @ v_var - grad_q

        mass_matrices.append(mass_matrix)
        biases.append(bias)

    dim = q.shape[-1]
    return ManipulatorTerms(
        mass_matrix=torch.stack(mass_matrices, dim=0).reshape(*sample_shape, dim, dim),
        bias_forces=torch.stack(biases, dim=0).reshape(*sample_shape, dim),
    )


def mass_matrix(
    system: LagrangianSystem,
    q: Tensor,
    qdot: Tensor | None = None,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Return the manipulator mass matrix ``M``."""
    if qdot is None:
        qdot = _zero_velocity_like(q)
    return manipulator_terms(system, q, qdot, create_graph=create_graph).mass_matrix


def bias_forces(
    system: LagrangianSystem,
    q: Tensor,
    qdot: Tensor,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Return all non-inertial generalized forces in ``M qddot + bias = tau``."""
    return manipulator_terms(system, q, qdot, create_graph=create_graph).bias_forces


def gravity_forces(
    system: LagrangianSystem,
    q: Tensor,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Return the zero-velocity generalized gravity/potential forces."""
    return bias_forces(system, q, _zero_velocity_like(q), create_graph=create_graph)


def velocity_forces(
    system: LagrangianSystem,
    q: Tensor,
    qdot: Tensor,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Return velocity-dependent bias forces with gravity removed."""
    return bias_forces(system, q, qdot, create_graph=create_graph) - gravity_forces(
        system,
        q,
        create_graph=create_graph,
    )


def inverse_dynamics(
    system: LagrangianSystem,
    q: Tensor,
    qdot: Tensor,
    qddot: Tensor,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Return generalized forces ``tau`` for a desired acceleration."""
    if qddot.shape != q.shape:
        raise ValueError("qddot must have the same shape as q")
    terms = manipulator_terms(system, q, qdot, create_graph=create_graph)
    inertial = (terms.mass_matrix @ qddot.unsqueeze(-1)).squeeze(-1)
    return inertial + terms.bias_forces


def forward_dynamics(
    system: LagrangianSystem,
    q: Tensor,
    qdot: Tensor,
    tau: Tensor,
    *,
    input_matrix: Tensor | None = None,
    create_graph: bool = False,
) -> Tensor:
    """Return ``qddot`` from generalized forces ``tau``."""
    terms = manipulator_terms(system, q, qdot, create_graph=create_graph)
    tau = _map_control_to_forces(tau, input_matrix, terms.bias_forces)
    rhs = tau - terms.bias_forces
    return torch.linalg.solve(terms.mass_matrix, rhs.unsqueeze(-1)).squeeze(-1)


def lagrangian_state_dynamics(
    system: LagrangianSystem,
    *,
    dim: int | None = None,
    input_matrix: Tensor | None = None,
    create_graph: bool = True,
):
    """Adapt a Lagrangian system into ``x' = f(t, x, u)`` over ``x = [q, qdot]``.

    If ``input_matrix`` is supplied with shape ``(dim, control_dim)``,
    controls are mapped to generalized forces with ``tau = B @ u``.
    """

    def dynamics(time: Tensor, state: Tensor, control: Tensor | None = None) -> Tensor:
        del time
        q, qdot = split_state(state, dim)
        if control is None:
            tau = torch.zeros_like(q)
        else:
            tau = control.to(dtype=state.dtype, device=state.device)
        qddot = forward_dynamics(
            system,
            q,
            qdot,
            tau,
            input_matrix=input_matrix,
            create_graph=create_graph,
        )
        return join_state(qdot, qddot)

    return dynamics


def _apply_gain(gain: float | Tensor, error: Tensor) -> Tensor:
    gain_tensor = torch.as_tensor(gain, dtype=error.dtype, device=error.device)
    if gain_tensor.ndim >= 2:
        return gain_tensor @ error if error.ndim == 1 else error @ gain_tensor.T
    return gain_tensor * error


def computed_torque(
    system: LagrangianSystem,
    q: Tensor,
    qdot: Tensor,
    q_desired: Tensor,
    *,
    qdot_desired: Tensor | None = None,
    qddot_desired: Tensor | None = None,
    kp: float | Tensor = 0.0,
    kd: float | Tensor = 0.0,
    create_graph: bool = False,
) -> Tensor:
    """Return inverse-dynamics torque with PD acceleration feedback."""
    if qdot_desired is None:
        qdot_desired = torch.zeros_like(qdot)
    if qddot_desired is None:
        qddot_desired = torch.zeros_like(q)

    position_error = q_desired - q
    velocity_error = qdot_desired - qdot
    commanded_acceleration = (
        qddot_desired
        + _apply_gain(kp, position_error)
        + _apply_gain(kd, velocity_error)
    )
    return inverse_dynamics(
        system,
        q,
        qdot,
        commanded_acceleration,
        create_graph=create_graph,
    )
