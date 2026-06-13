"""Analytical mechanics systems built on Torch autograd."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

Tensor = torch.Tensor
ScalarFn2 = Callable[[Tensor, Tensor], Tensor]
ScalarFn1 = Callable[[Tensor], Tensor]
VectorFn1 = Callable[[Tensor], Tensor]
TransformFn = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]


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
    if not output.requires_grad:
        return tuple(torch.zeros_like(inp) for inp in inputs)
    grads = torch.autograd.grad(
        output,
        inputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return tuple(torch.zeros_like(inp) if grad is None else grad for inp, grad in zip(inputs, grads))


def _vector_like(value: Tensor, reference: Tensor, name: str) -> Tensor:
    if value.shape != reference.shape:
        raise ValueError(
            f"{name} must return shape {tuple(reference.shape)}, got {tuple(value.shape)}"
        )
    return value


def _symplectic_matrix(dim: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    omega = torch.zeros((2 * dim, 2 * dim), dtype=dtype, device=device)
    eye = torch.eye(dim, dtype=dtype, device=device)
    omega[:dim, dim:] = eye
    omega[dim:, :dim] = -eye
    return omega


def canonical_transformation_residual(
    transform_fn: TransformFn,
    q: Tensor,
    p: Tensor,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Return ``J.T @ omega @ J - omega`` for a phase-space transform.

    A zero matrix means the local transformation preserves the canonical
    symplectic form for the supplied samples.
    """
    (q_flat, p_flat), sample_shape = _flatten_state(q, p)
    dim = q.shape[-1]
    residuals = []

    for q_i, p_i in zip(q_flat, p_flat):
        z_var = torch.cat([q_i, p_i]).clone().requires_grad_(True)
        q_var = z_var[:dim]
        p_var = z_var[dim:]
        next_q, next_p = transform_fn(q_var, p_var)
        next_q = _vector_like(next_q, q_var, "transform_fn q")
        next_p = _vector_like(next_p, p_var, "transform_fn p")
        transformed = torch.cat([next_q, next_p])

        jacobian_rows = []
        for component in transformed:
            (grad_z,) = _safe_grad(component, (z_var,), create_graph=create_graph)
            jacobian_rows.append(grad_z)

        jacobian = torch.stack(jacobian_rows, dim=0)
        omega = _symplectic_matrix(dim, dtype=q.dtype, device=q.device)
        residuals.append(jacobian.T @ omega @ jacobian - omega)

    return torch.stack(residuals, dim=0).reshape(*sample_shape, 2 * dim, 2 * dim)


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

    def momentum(self, q: Tensor, qdot: Tensor, *, create_graph: bool = False) -> Tensor:
        """Return generalized momenta ``dL/dqdot``."""
        (q_flat, qdot_flat), sample_shape = _flatten_state(q, qdot)
        momenta = []

        for q_i, v_i in zip(q_flat, qdot_flat):
            q_var = q_i.clone().requires_grad_(True)
            v_var = v_i.clone().requires_grad_(True)
            lagrangian = self.lagrangian(q_var, v_var)
            (grad_v,) = _safe_grad(
                lagrangian,
                (v_var,),
                create_graph=create_graph,
            )
            momenta.append(grad_v)

        return torch.stack(momenta, dim=0).reshape(*sample_shape, q.shape[-1])

    def noether_charge(
        self,
        q: Tensor,
        qdot: Tensor,
        generator: VectorFn1,
        *,
        create_graph: bool = False,
    ) -> Tensor:
        """Return ``p . xi(q)`` for an infinitesimal coordinate generator."""
        momentum = self.momentum(q, qdot, create_graph=create_graph)
        (q_flat, momentum_flat), sample_shape = _flatten_state(q, momentum)
        charges = []

        for q_i, p_i in zip(q_flat, momentum_flat):
            xi = _vector_like(generator(q_i), q_i, "generator")
            charges.append(torch.dot(p_i, xi))

        return torch.stack(charges, dim=0).reshape(sample_shape)

    def coordinate_symmetry_residual(
        self,
        q: Tensor,
        qdot: Tensor,
        generator: VectorFn1,
        *,
        create_graph: bool = False,
    ) -> Tensor:
        """Return the infinitesimal variation of ``L`` under ``delta q = xi(q)``."""
        (q_flat, qdot_flat), sample_shape = _flatten_state(q, qdot)
        residuals = []

        for q_i, v_i in zip(q_flat, qdot_flat):
            q_var = q_i.clone().requires_grad_(True)
            v_var = v_i.clone().requires_grad_(True)
            lagrangian = self.lagrangian(q_var, v_var)
            grad_q, grad_v = _safe_grad(
                lagrangian,
                (q_var, v_var),
                create_graph=create_graph,
            )
            xi = _vector_like(generator(q_var), q_var, "generator")

            generator_rows = []
            for component in xi:
                (grad_xi,) = _safe_grad(component, (q_var,), create_graph=create_graph)
                generator_rows.append(grad_xi)

            generator_jacobian = torch.stack(generator_rows, dim=0)
            delta_qdot = generator_jacobian @ v_var
            residuals.append(torch.dot(grad_q, xi) + torch.dot(grad_v, delta_qdot))

        return torch.stack(residuals, dim=0).reshape(sample_shape)

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

    def poisson_bracket(
        self,
        left: ScalarFn2,
        right: ScalarFn2,
        q: Tensor,
        p: Tensor,
        *,
        create_graph: bool = False,
    ) -> Tensor:
        """Return the canonical Poisson bracket ``{left, right}``."""
        (q_flat, p_flat), sample_shape = _flatten_state(q, p)
        brackets = []

        for q_i, p_i in zip(q_flat, p_flat):
            q_var = q_i.clone().requires_grad_(True)
            p_var = p_i.clone().requires_grad_(True)
            left_value = _scalar(left(q_var, p_var), "left")
            right_value = _scalar(right(q_var, p_var), "right")
            left_q, left_p = _safe_grad(
                left_value,
                (q_var, p_var),
                create_graph=create_graph,
                retain_graph=True,
            )
            right_q, right_p = _safe_grad(
                right_value,
                (q_var, p_var),
                create_graph=create_graph,
                retain_graph=True,
            )
            brackets.append(torch.dot(left_q, right_p) - torch.dot(left_p, right_q))

        return torch.stack(brackets, dim=0).reshape(sample_shape)

    def time_derivative(
        self,
        observable: ScalarFn2,
        q: Tensor,
        p: Tensor,
        *,
        create_graph: bool = False,
    ) -> Tensor:
        """Return ``d observable / dt`` from its Poisson bracket with ``H``."""
        return self.poisson_bracket(
            observable,
            self.hamiltonian,
            q,
            p,
            create_graph=create_graph,
        )

    def energy(self, q: Tensor, p: Tensor) -> Tensor:
        """Evaluate the Hamiltonian over a batch."""
        (q_flat, p_flat), sample_shape = _flatten_state(q, p)
        values = []
        for q_i, p_i in zip(q_flat, p_flat):
            values.append(self.hamiltonian(q_i, p_i))
        return torch.stack(values, dim=0).reshape(sample_shape)
