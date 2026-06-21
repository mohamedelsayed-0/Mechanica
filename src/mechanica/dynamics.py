"""State-space dynamics utilities for controls, robotics, and learning."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import inspect

import torch

from .analytical import HamiltonianSystem

Tensor = torch.Tensor
StateDynamics = Callable[..., Tensor]
AccelerationFn = Callable[..., Tensor]


def join_state(position: Tensor, velocity: Tensor) -> Tensor:
    """Return a concatenated second-order state ``[..., q, qdot]``."""
    if position.shape[:-1] != velocity.shape[:-1]:
        raise ValueError("position and velocity must have the same sample shape")
    return torch.cat([position, velocity], dim=-1)


def split_state(state: Tensor, dim: int | None = None) -> tuple[Tensor, Tensor]:
    """Split ``[..., q, qdot]`` or ``[..., q, p]`` into two vector halves."""
    state_dim = state.shape[-1]
    if dim is None:
        if state_dim % 2 != 0:
            raise ValueError("state final dimension must be even when dim is omitted")
        dim = state_dim // 2
    if dim <= 0 or dim >= state_dim:
        raise ValueError("dim must split the state final dimension into two nonempty parts")
    return state[..., :dim], state[..., dim:]


def _call_accepts_arg(fn: Callable[..., Tensor], count: int) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return True

    positional = [
        param
        for param in params
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    ]
    if any(param.kind == param.VAR_POSITIONAL for param in params):
        return True
    return len(positional) >= count


def _call_dynamics(
    dynamics: StateDynamics,
    time: Tensor,
    state: Tensor,
    control: Tensor | None,
) -> Tensor:
    if _call_accepts_arg(dynamics, 3):
        return dynamics(time, state, control)
    if control is not None:
        raise ValueError("controls were provided but dynamics does not accept a control argument")
    return dynamics(time, state)


def _call_acceleration(
    acceleration_fn: AccelerationFn,
    time: Tensor,
    position: Tensor,
    velocity: Tensor,
    control: Tensor | None,
) -> Tensor:
    if _call_accepts_arg(acceleration_fn, 4):
        return acceleration_fn(time, position, velocity, control)
    if _call_accepts_arg(acceleration_fn, 3):
        return acceleration_fn(position, velocity, control)
    if control is not None:
        raise ValueError("controls were provided but acceleration_fn does not accept control")
    return acceleration_fn(position, velocity)


def second_order_dynamics(
    acceleration_fn: AccelerationFn,
    *,
    dim: int | None = None,
) -> StateDynamics:
    """Wrap ``qddot = a(t, q, qdot, u)`` as state dynamics ``x' = f(t, x, u)``."""

    def dynamics(time: Tensor, state: Tensor, control: Tensor | None = None) -> Tensor:
        position, velocity = split_state(state, dim)
        acceleration = _call_acceleration(acceleration_fn, time, position, velocity, control)
        if acceleration.shape != position.shape:
            raise ValueError(
                f"acceleration_fn must return shape {tuple(position.shape)}, "
                f"got {tuple(acceleration.shape)}"
            )
        return join_state(velocity, acceleration)

    return dynamics


def hamiltonian_dynamics(
    system: HamiltonianSystem,
    *,
    dim: int | None = None,
    input_matrix: Tensor | None = None,
    create_graph: bool = True,
) -> StateDynamics:
    """Adapt a Hamiltonian system into ``x' = f(t, x, u)`` over ``x = [q, p]``.

    Controls are interpreted as generalized forces added to ``pdot``. If
    ``input_matrix`` is supplied with shape ``(dim, control_dim)``, controls are
    first mapped through that matrix.
    """

    def dynamics(time: Tensor, state: Tensor, control: Tensor | None = None) -> Tensor:
        del time
        q, p = split_state(state, dim)
        dqdt, dpdt = system.vector_field(q, p, create_graph=create_graph)

        if control is not None:
            force = control
            if input_matrix is not None:
                matrix = input_matrix.to(dtype=control.dtype, device=control.device)
                force = matrix @ control if control.ndim == 1 else control @ matrix.T
            if force.shape != dpdt.shape:
                raise ValueError(
                    f"control force must have shape {tuple(dpdt.shape)}, got {tuple(force.shape)}"
                )
            dpdt = dpdt + force

        return join_state(dqdt, dpdt)

    return dynamics


def _control_at(
    controls: Tensor | Sequence[Tensor] | None,
    index: int,
) -> Tensor | None:
    if controls is None:
        return None
    return controls[index]


def euler_state_step(
    dynamics: StateDynamics,
    time: Tensor,
    state: Tensor,
    dt: Tensor,
    control: Tensor | None = None,
) -> Tensor:
    """One explicit Euler step for state-space dynamics."""
    return state + dt * _call_dynamics(dynamics, time, state, control)


def rk4_state_step(
    dynamics: StateDynamics,
    time: Tensor,
    state: Tensor,
    dt: Tensor,
    control: Tensor | None = None,
) -> Tensor:
    """One fourth-order Runge-Kutta step for state-space dynamics."""
    half = dt / 2
    k1 = _call_dynamics(dynamics, time, state, control)
    k2 = _call_dynamics(dynamics, time + half, state + half * k1, control)
    k3 = _call_dynamics(dynamics, time + half, state + half * k2, control)
    k4 = _call_dynamics(dynamics, time + dt, state + dt * k3, control)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rollout(
    dynamics: StateDynamics,
    initial_state: Tensor,
    times: Tensor,
    *,
    controls: Tensor | Sequence[Tensor] | None = None,
    method: str = "rk4",
) -> Tensor:
    """Integrate state-space dynamics over a 1D time grid.

    ``controls`` are held constant over each interval and should have length
    ``len(times) - 1`` when provided.
    """
    if times.ndim != 1:
        raise ValueError("times must be a one-dimensional tensor")
    if times.numel() == 0:
        raise ValueError("at least one time sample is required")

    state = initial_state
    states = [state]
    for index in range(times.numel() - 1):
        time = times[index].to(dtype=state.dtype, device=state.device)
        dt = (times[index + 1] - times[index]).to(dtype=state.dtype, device=state.device)
        control = _control_at(controls, index)
        if control is not None:
            control = control.to(dtype=state.dtype, device=state.device)

        if method == "euler":
            state = euler_state_step(dynamics, time, state, dt, control)
        elif method == "rk4":
            state = rk4_state_step(dynamics, time, state, dt, control)
        else:
            raise ValueError("method must be either 'euler' or 'rk4'")
        states.append(state)

    return torch.stack(states, dim=0)


def _jacobian(output: Tensor, inputs: Tensor, *, create_graph: bool) -> Tensor:
    rows = []
    flat_output = output.reshape(-1)
    for component in flat_output:
        if component.requires_grad:
            (grad,) = torch.autograd.grad(
                component,
                inputs,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True,
            )
        else:
            grad = None
        if grad is None:
            grad = torch.zeros_like(inputs)
        rows.append(grad.reshape(-1))
    return torch.stack(rows, dim=0)


def linearize(
    dynamics: StateDynamics,
    time: float | Tensor,
    state: Tensor,
    control: Tensor | None = None,
    *,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return continuous-time Jacobians ``A = df/dx`` and ``B = df/du``."""
    if state.ndim != 1:
        raise ValueError("linearize expects an unbatched one-dimensional state")

    state_var = state.clone().detach().requires_grad_(True)
    time_tensor = torch.as_tensor(time, dtype=state.dtype, device=state.device)

    control_var = None
    if control is not None:
        if control.ndim != 1:
            raise ValueError("linearize expects an unbatched one-dimensional control")
        control_var = control.clone().detach().requires_grad_(True)

    value = _call_dynamics(dynamics, time_tensor, state_var, control_var)
    if value.shape != state.shape:
        raise ValueError(f"dynamics must return shape {tuple(state.shape)}, got {tuple(value.shape)}")

    state_jacobian = _jacobian(value, state_var, create_graph=create_graph)
    if control_var is None:
        control_jacobian = torch.zeros(
            state.numel(),
            0,
            dtype=state.dtype,
            device=state.device,
        )
    else:
        control_jacobian = _jacobian(value, control_var, create_graph=create_graph)

    return state_jacobian, control_jacobian
