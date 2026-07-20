"""Differentiable trajectory optimization for robot dynamics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

Tensor = torch.Tensor
DiscreteDynamics = Callable[[Tensor, Tensor], Tensor]
StageCost = Callable[[Tensor, Tensor, int], Tensor]
TerminalCost = Callable[[Tensor], Tensor]


@dataclass
class ILQRResult:
    states: Tensor
    controls: Tensor
    cost_history: Tensor
    converged: bool

    @property
    def final_cost(self) -> float:
        return float(self.cost_history[..., -1].mean().detach().cpu())


def _rollout(dynamics: DiscreteDynamics, initial_state: Tensor, controls: Tensor) -> Tensor:
    states = [initial_state]
    for control in controls:
        states.append(dynamics(states[-1], control))
    return torch.stack(states)


def _total_cost(states: Tensor, controls: Tensor, cost: StageCost, terminal: TerminalCost) -> Tensor:
    return sum(cost(states[index], control, index) for index, control in enumerate(controls)) + terminal(states[-1])


def _single_ilqr(
    dynamics: DiscreteDynamics,
    initial_state: Tensor,
    initial_controls: Tensor,
    cost: StageCost,
    terminal_cost: TerminalCost,
    *,
    method: str,
    max_iterations: int,
    tolerance: float,
    regularization: float,
    control_limits: tuple[Tensor, Tensor] | None,
) -> ILQRResult:
    if method not in {"ilqr", "ddp"}:
        raise ValueError("method must be 'ilqr' or 'ddp'")
    controls = initial_controls
    states = _rollout(dynamics, initial_state, controls)
    history = [_total_cost(states, controls, cost, terminal_cost)]
    converged = False

    for _ in range(max_iterations):
        final = states[-1].detach().requires_grad_(True)
        value_gradient = torch.autograd.functional.jacobian(terminal_cost, final)
        value_hessian = torch.autograd.functional.hessian(terminal_cost, final)
        feedforward: list[Tensor] = []
        feedback: list[Tensor] = []

        for step in range(controls.shape[0] - 1, -1, -1):
            state = states[step].detach()
            control = controls[step].detach()
            state_dim = state.numel()
            z = torch.cat((state, control)).requires_grad_(True)

            def local_cost(value: Tensor) -> Tensor:
                return cost(value[:state_dim], value[state_dim:], step)

            def local_dynamics(value: Tensor) -> Tensor:
                return dynamics(value[:state_dim], value[state_dim:])

            gradient = torch.autograd.functional.jacobian(local_cost, z)
            hessian = torch.autograd.functional.hessian(local_cost, z)
            dynamics_jacobian = torch.autograd.functional.jacobian(local_dynamics, z)
            q_gradient = gradient + dynamics_jacobian.T @ value_gradient
            q_hessian = hessian + dynamics_jacobian.T @ value_hessian @ dynamics_jacobian
            if method == "ddp":
                q_hessian = q_hessian + torch.autograd.functional.hessian(
                    lambda value: (local_dynamics(value) * value_gradient).sum(), z
                )
            q_hessian = 0.5 * (q_hessian + q_hessian.T)
            qx, qu = q_gradient[:state_dim], q_gradient[state_dim:]
            qxx = q_hessian[:state_dim, :state_dim]
            qux = q_hessian[state_dim:, :state_dim]
            quu = q_hessian[state_dim:, state_dim:]
            quu = quu + regularization * torch.eye(quu.shape[0], dtype=z.dtype, device=z.device)
            gain = -torch.linalg.solve(quu, qux)
            change = -torch.linalg.solve(quu, qu)
            feedforward.append(change)
            feedback.append(gain)
            value_gradient = qx + gain.T @ quu @ change + gain.T @ qu + qux.T @ change
            value_hessian = qxx + gain.T @ quu @ gain + gain.T @ qux + qux.T @ gain
            value_hessian = 0.5 * (value_hessian + value_hessian.T)

        feedforward.reverse()
        feedback.reverse()
        accepted = None
        for scale in (1.0, 0.5, 0.25, 0.1, 0.05):
            candidate_states = [initial_state]
            candidate_controls = []
            for step, nominal_control in enumerate(controls):
                control = nominal_control + scale * feedforward[step]
                control = control + feedback[step] @ (candidate_states[-1] - states[step])
                if control_limits is not None:
                    control = control.clamp(control_limits[0].to(control), control_limits[1].to(control))
                candidate_controls.append(control)
                candidate_states.append(dynamics(candidate_states[-1], control))
            candidate_states_tensor = torch.stack(candidate_states)
            candidate_controls_tensor = torch.stack(candidate_controls)
            candidate_cost = _total_cost(candidate_states_tensor, candidate_controls_tensor, cost, terminal_cost)
            if candidate_cost < history[-1]:
                accepted = candidate_states_tensor, candidate_controls_tensor, candidate_cost
                break
        if accepted is None:
            break
        states, controls, next_cost = accepted
        improvement = history[-1] - next_cost
        history.append(next_cost)
        if improvement.detach().abs().item() <= tolerance:
            converged = True
            break
    return ILQRResult(states, controls, torch.stack(history), converged)


def ilqr(
    dynamics: DiscreteDynamics,
    initial_state: Tensor,
    initial_controls: Tensor,
    cost: StageCost,
    terminal_cost: TerminalCost,
    *,
    method: str = "ilqr",
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    regularization: float = 1e-6,
    control_limits: tuple[Tensor, Tensor] | None = None,
) -> ILQRResult:
    """Optimize controls with batched iLQR or second-order DDP."""
    if initial_state.ndim == 1:
        return _single_ilqr(
            dynamics, initial_state, initial_controls, cost, terminal_cost,
            method=method, max_iterations=max_iterations, tolerance=tolerance,
            regularization=regularization, control_limits=control_limits,
        )
    results = [
        _single_ilqr(
            dynamics, state, controls, cost, terminal_cost,
            method=method, max_iterations=max_iterations, tolerance=tolerance,
            regularization=regularization, control_limits=control_limits,
        )
        for state, controls in zip(initial_state, initial_controls)
    ]
    length = max(result.cost_history.numel() for result in results)
    histories = [
        torch.cat((result.cost_history, result.cost_history[-1].expand(length - result.cost_history.numel())))
        for result in results
    ]
    return ILQRResult(
        torch.stack([result.states for result in results]),
        torch.stack([result.controls for result in results]),
        torch.stack(histories),
        all(result.converged for result in results),
    )


def direct_collocation_residual(
    dynamics: Callable[[Tensor, Tensor, Tensor], Tensor],
    states: Tensor,
    controls: Tensor,
    times: Tensor,
) -> Tensor:
    """Return trapezoidal direct-collocation defects over a trajectory."""
    if states.shape[0] != times.numel() or controls.shape[0] != times.numel() - 1:
        raise ValueError("states, controls, and times have inconsistent horizons")
    defects = []
    for step, control in enumerate(controls):
        dt = times[step + 1] - times[step]
        start = dynamics(times[step], states[step], control)
        end = dynamics(times[step + 1], states[step + 1], control)
        defects.append(states[step + 1] - states[step] - 0.5 * dt * (start + end))
    return torch.stack(defects)


def mpc_control(*args, **kwargs) -> tuple[Tensor, ILQRResult]:
    """Run iLQR/DDP and return the first receding-horizon control."""
    result = ilqr(*args, **kwargs)
    return result.controls[..., 0, :], result
