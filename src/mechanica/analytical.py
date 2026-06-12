"""Analytical mechanics systems built on Torch autograd."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

Tensor = torch.Tensor
ScalarFn2 = Callable[[Tensor, Tensor], Tensor]
ScalarFn1 = Callable[[Tensor], Tensor]


def _scalar(value: Tensor, name: str) -> Tensor:
    if value.ndim == 0:
        return value
    if value.numel() == 1:
        return value.reshape(())
    raise ValueError(f"{name} must return a scalar tensor, got shape {tuple(value.shape)}")


def _flatten_state(*states: Tensor) -> tuple[list[Tensor], torch.Size]:
    if not states:
        raise ValueError("at least one state tensor is required")
    dim = states[0].shape[-1]
    sample_shape = states[0].shape[:-1]
    flat = []
    for state in states:
        if state.shape[-1] != dim:
            raise ValueError("all state tensors must have the same final dimension")
        if state.shape[:-1] != sample_shape:
            raise ValueError("all state tensors must have the same sample shape")
        flat.append(state.reshape(-1, dim))
    return flat, sample_shape


def _safe_grad(
    output: Tensor,
    inputs: tuple[Tensor, ...],
    *,
    create_graph: bool,
    retain_graph: bool = True,
) -> tuple[Tensor, ...]:
    grads = torch.autograd.grad(
        output,
        inputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return tuple(torch.zeros_like(inp) if grad is None else grad for inp, grad in zip(inputs, grads))


@dataclass
class LagrangianSystem:
    """A time-independent Lagrangian system.

    The system can be supplied either as a full Lagrangian ``L(q, qdot)`` or as
    separate kinetic and potential energy callables. Inputs use the final tensor
    dimension as the generalized coordinate dimension.
    """

    kinetic: ScalarFn2 | None = None
    potential: ScalarFn1 | None = None
    lagrangian_fn: ScalarFn2 | None = None
    name: str = "lagrangian_system"

    def __post_init__(self) -> None:
        if self.lagrangian_fn is None and self.kinetic is None:
            raise ValueError("provide either lagrangian_fn or kinetic")

    def lagrangian(self, q: Tensor, qdot: Tensor) -> Tensor:
        """Evaluate the scalar Lagrangian for one sample."""
        if self.lagrangian_fn is not None:
            return _scalar(self.lagrangian_fn(q, qdot), "lagrangian_fn")

        kinetic = _scalar(self.kinetic(q, qdot), "kinetic")  # type: ignore[misc]
        potential = torch.zeros((), dtype=q.dtype, device=q.device)
        if self.potential is not None:
            potential = _scalar(self.potential(q), "potential")
        return kinetic - potential

    def euler_lagrange_residual(
        self,
        q: Tensor,
        qdot: Tensor,
        qddot: Tensor,
        *,
        create_graph: bool = False,
    ) -> Tensor:
        """Return ``d/dt(dL/dqdot) - dL/dq`` evaluated on a trajectory.

        For a time-independent Lagrangian, the total derivative is expanded as
        ``L_vv @ qddot + L_vq @ qdot``. This keeps the method local in time and
        makes it useful for trajectory fitting and inverse mechanics.
        """
        (q_flat, qdot_flat, qddot_flat), sample_shape = _flatten_state(q, qdot, qddot)
        residuals = []

        for q_i, v_i, a_i in zip(q_flat, qdot_flat, qddot_flat):
            q_var = q_i.clone().requires_grad_(True)
            v_var = v_i.clone().requires_grad_(True)
            lagrangian = self.lagrangian(q_var, v_var)

            grad_q, grad_v = _safe_grad(
                lagrangian,
                (q_var, v_var),
                create_graph=True,
            )

            h_vq_rows = []
            h_vv_rows = []
            for component in grad_v:
                h_vq_i, h_vv_i = _safe_grad(
                    component,
                    (q_var, v_var),
                    create_graph=create_graph,
                )
                h_vq_rows.append(h_vq_i)
                h_vv_rows.append(h_vv_i)

            h_vq = torch.stack(h_vq_rows, dim=0)
            h_vv = torch.stack(h_vv_rows, dim=0)
            residual = h_vv @ a_i + h_vq @ v_i - grad_q
            residuals.append(residual)

        return torch.stack(residuals, dim=0).reshape(*sample_shape, q.shape[-1])

    def energy(self, q: Tensor, qdot: Tensor, *, create_graph: bool = False) -> Tensor:
        """Return the generalized energy ``qdot . dL/dqdot - L``."""
        (q_flat, qdot_flat), sample_shape = _flatten_state(q, qdot)
        energies = []

        for q_i, v_i in zip(q_flat, qdot_flat):
            q_var = q_i.clone().requires_grad_(True)
            v_var = v_i.clone().requires_grad_(True)
            lagrangian = self.lagrangian(q_var, v_var)
            (grad_v,) = _safe_grad(
                lagrangian,
                (v_var,),
                create_graph=create_graph,
            )
            energies.append(torch.dot(v_var, grad_v) - lagrangian)

        return torch.stack(energies, dim=0).reshape(sample_shape)


@dataclass
class HamiltonianSystem:
    """A Hamiltonian system with canonical coordinates ``(q, p)``."""

    hamiltonian_fn: ScalarFn2
    name: str = "hamiltonian_system"

    def hamiltonian(self, q: Tensor, p: Tensor) -> Tensor:
        """Evaluate the scalar Hamiltonian for one sample."""
        return _scalar(self.hamiltonian_fn(q, p), "hamiltonian_fn")

    def vector_field(
        self,
        q: Tensor,
        p: Tensor,
        *,
        create_graph: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(dq/dt, dp/dt) = (dH/dp, -dH/dq)``."""
        (q_flat, p_flat), sample_shape = _flatten_state(q, p)
        dqdt = []
        dpdt = []

        for q_i, p_i in zip(q_flat, p_flat):
            q_var = q_i.clone().requires_grad_(True)
            p_var = p_i.clone().requires_grad_(True)
            hamiltonian = self.hamiltonian(q_var, p_var)
            grad_q, grad_p = _safe_grad(
                hamiltonian,
                (q_var, p_var),
                create_graph=create_graph,
            )
            dqdt.append(grad_p)
            dpdt.append(-grad_q)

        dim = q.shape[-1]
        return (
            torch.stack(dqdt, dim=0).reshape(*sample_shape, dim),
            torch.stack(dpdt, dim=0).reshape(*sample_shape, dim),
        )

    def hamilton_equations_residual(
        self,
        q: Tensor,
        p: Tensor,
        qdot: Tensor,
        pdot: Tensor,
        *,
        create_graph: bool = False,
    ) -> Tensor:
        """Return concatenated Hamilton equation residuals."""
        dqdt, dpdt = self.vector_field(q, p, create_graph=create_graph)
        return torch.cat([qdot - dqdt, pdot - dpdt], dim=-1)

    def energy(self, q: Tensor, p: Tensor) -> Tensor:
        """Evaluate the Hamiltonian over a batch."""
        (q_flat, p_flat), sample_shape = _flatten_state(q, p)
        values = []
        for q_i, p_i in zip(q_flat, p_flat):
            values.append(self.hamiltonian(q_i, p_i))
        return torch.stack(values, dim=0).reshape(sample_shape)
