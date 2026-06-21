"""Control helpers built on Mechanica state-space dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dynamics import StateDynamics, discrete_step, linearize, rollout

Tensor = torch.Tensor


@dataclass
class LQRResult:
    """Finite-horizon discrete LQR solution.

    The feedback law is ``u = u_ref - K @ (x - x_ref)`` when nominal
    trajectories are present, and ``u = -K @ x`` otherwise.
    """

    gains: Tensor
    value_matrices: Tensor
    discrete_A: Tensor | None = None
    discrete_B: Tensor | None = None
    nominal_states: Tensor | None = None
    nominal_controls: Tensor | None = None
    iterations: int | None = None
    converged: bool | None = None

    def _step_index(self, step: int) -> int:
        if self.gains.shape[0] == 1:
            return 0
        return min(step, self.gains.shape[0] - 1)

    def control(
        self,
        state: Tensor,
        step: int,
        *,
        target_state: Tensor | None = None,
        nominal_control: Tensor | None = None,
    ) -> Tensor:
        """Return the feedback control for ``state`` at a horizon step."""
        index = self._step_index(step)
        gain = self.gains[index]
        reference = target_state
        if reference is None and self.nominal_states is not None:
            reference = self.nominal_states[min(step, self.nominal_states.shape[0] - 1)]
        if reference is None:
            error = state
        else:
            error = state - reference.to(dtype=state.dtype, device=state.device)

        correction = gain @ error if error.ndim == 1 else error @ gain.T

        nominal = nominal_control
        if nominal is None and self.nominal_controls is not None:
            nominal = self.nominal_controls[min(step, self.nominal_controls.shape[0] - 1)]
        if nominal is None:
            return -correction
        return nominal.to(dtype=state.dtype, device=state.device) - correction


def _as_horizon_matrix(matrix: Tensor, horizon: int, name: str) -> Tensor:
    if matrix.ndim == 2:
        return matrix.unsqueeze(0).expand(horizon, -1, -1)
    if matrix.ndim == 3 and matrix.shape[0] == horizon:
        return matrix
    raise ValueError(f"{name} must be a matrix or a horizon-length matrix sequence")


def finite_horizon_lqr(
    A: Tensor,
    B: Tensor,
    Q: Tensor,
    R: Tensor,
    Qf: Tensor | None = None,
    *,
    horizon: int | None = None,
    regularization: float = 1e-9,
) -> LQRResult:
    """Solve the discrete finite-horizon LQR problem by Riccati recursion.

    ``A`` and ``B`` may be constant matrices or time-varying sequences with
    leading dimension ``horizon``. The returned gains use ``u = -K x``.
    """
    if horizon is None:
        if A.ndim == 3:
            horizon = A.shape[0]
        elif B.ndim == 3:
            horizon = B.shape[0]
        else:
            raise ValueError("horizon is required for constant A and B")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    a_seq = _as_horizon_matrix(A, horizon, "A")
    b_seq = _as_horizon_matrix(B, horizon, "B")
    q_seq = _as_horizon_matrix(Q, horizon, "Q")
    r_seq = _as_horizon_matrix(R, horizon, "R")

    n = a_seq.shape[-1]
    m = b_seq.shape[-1]
    if a_seq.shape[-2:] != (n, n):
        raise ValueError("A must be square")
    if b_seq.shape[-2] != n:
        raise ValueError("B row dimension must match A")
    if q_seq.shape[-2:] != (n, n):
        raise ValueError("Q shape must match state dimension")
    if r_seq.shape[-2:] != (m, m):
        raise ValueError("R shape must match control dimension")

    terminal = Qf if Qf is not None else (Q[-1] if Q.ndim == 3 else Q)
    terminal = terminal.to(dtype=A.dtype, device=A.device)

    gains = []
    values = [terminal]
    value = terminal
    identity = torch.eye(m, dtype=A.dtype, device=A.device)

    for step in range(horizon - 1, -1, -1):
        a = a_seq[step]
        b = b_seq[step]
        q = q_seq[step]
        r = r_seq[step]

        control_hessian = r + b.T @ value @ b
        if regularization:
            control_hessian = control_hessian + regularization * identity
        gain = torch.linalg.solve(control_hessian, b.T @ value @ a)
        value = q + a.T @ value @ (a - b @ gain)

        gains.append(gain)
        values.append(value)

    gains.reverse()
    values.reverse()
    return LQRResult(
        gains=torch.stack(gains, dim=0),
        value_matrices=torch.stack(values, dim=0),
        discrete_A=a_seq,
        discrete_B=b_seq,
    )


def infinite_horizon_lqr(
    A: Tensor,
    B: Tensor,
    Q: Tensor,
    R: Tensor,
    *,
    tolerance: float = 1e-7,
    max_iterations: int = 1000,
    regularization: float = 1e-9,
) -> LQRResult:
    """Solve the discrete infinite-horizon LQR problem by Riccati iteration."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("B row dimension must match A")
    if Q.shape != A.shape:
        raise ValueError("Q shape must match A")
    control_dim = B.shape[1]
    if R.shape != (control_dim, control_dim):
        raise ValueError("R shape must match control dimension")

    value = Q.to(dtype=A.dtype, device=A.device)
    identity = torch.eye(control_dim, dtype=A.dtype, device=A.device)
    gain = torch.zeros(control_dim, A.shape[0], dtype=A.dtype, device=A.device)
    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        control_hessian = R + B.T @ value @ B
        if regularization:
            control_hessian = control_hessian + regularization * identity
        gain = torch.linalg.solve(control_hessian, B.T @ value @ A)
        next_value = Q + A.T @ value @ (A - B @ gain)
        iterations = iteration
        scale = next_value.abs().max().clamp_min(1)
        if (next_value - value).abs().max() <= tolerance * scale:
            value = next_value
            converged = True
            break
        value = next_value

    return LQRResult(
        gains=gain.unsqueeze(0),
        value_matrices=value.unsqueeze(0),
        discrete_A=A,
        discrete_B=B,
        iterations=iterations,
        converged=converged,
    )


def tvlqr(
    dynamics: StateDynamics,
    times: Tensor,
    states: Tensor,
    controls: Tensor,
    Q: Tensor,
    R: Tensor,
    Qf: Tensor | None = None,
    *,
    regularization: float = 1e-9,
) -> LQRResult:
    """Build a time-varying LQR controller by linearizing a trajectory.

    Continuous-time Jacobians are discretized with a first-order hold:
    ``A_d = I + dt A_c`` and ``B_d = dt B_c``.
    """
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    horizon = times.numel() - 1
    if horizon <= 0:
        raise ValueError("at least two time samples are required")
    if states.shape[0] != times.numel():
        raise ValueError("states must have one sample for each time")
    if controls.shape[0] != horizon:
        raise ValueError("controls must have one sample per interval")

    discrete_a = []
    discrete_b = []
    for step in range(horizon):
        continuous_a, continuous_b = linearize(dynamics, times[step], states[step], controls[step])
        dt = (times[step + 1] - times[step]).to(dtype=states.dtype, device=states.device)
        identity = torch.eye(continuous_a.shape[0], dtype=states.dtype, device=states.device)
        discrete_a.append(identity + dt * continuous_a)
        discrete_b.append(dt * continuous_b)

    result = finite_horizon_lqr(
        torch.stack(discrete_a, dim=0),
        torch.stack(discrete_b, dim=0),
        Q,
        R,
        Qf,
        regularization=regularization,
    )
    result.nominal_states = states
    result.nominal_controls = controls
    return result


def quadratic_cost(
    states: Tensor,
    controls: Tensor,
    Q: Tensor,
    R: Tensor,
    Qf: Tensor | None = None,
    *,
    target_states: Tensor | None = None,
    target_controls: Tensor | None = None,
) -> Tensor:
    """Return a differentiable finite-horizon quadratic trajectory cost."""
    horizon = controls.shape[0]
    if states.shape[0] != horizon + 1:
        raise ValueError("states must have one more sample than controls")

    q_seq = _as_horizon_matrix(Q, horizon, "Q")
    r_seq = _as_horizon_matrix(R, horizon, "R")
    terminal = Qf if Qf is not None else (Q[-1] if Q.ndim == 3 else Q)

    state_error = states[:-1] if target_states is None else states[:-1] - target_states[:-1]
    control_error = controls if target_controls is None else controls - target_controls
    terminal_error = states[-1] if target_states is None else states[-1] - target_states[-1]

    running_state = torch.einsum("bi,bij,bj->b", state_error, q_seq, state_error).sum()
    running_control = torch.einsum("bi,bij,bj->b", control_error, r_seq, control_error).sum()
    terminal_cost = terminal_error @ terminal @ terminal_error
    return running_state + running_control + terminal_cost


def rollout_cost(
    dynamics: StateDynamics,
    initial_state: Tensor,
    times: Tensor,
    controls: Tensor,
    Q: Tensor,
    R: Tensor,
    Qf: Tensor | None = None,
    *,
    target_states: Tensor | None = None,
    target_controls: Tensor | None = None,
    method: str = "rk4",
) -> Tensor:
    """Roll out controlled dynamics and return quadratic trajectory cost."""
    states = rollout(
        dynamics,
        initial_state,
        times,
        controls=controls,
        method=method,
    )
    return quadratic_cost(
        states,
        controls,
        Q,
        R,
        Qf,
        target_states=target_states,
        target_controls=target_controls,
    )


def feedback_rollout(
    dynamics: StateDynamics,
    controller: LQRResult,
    initial_state: Tensor,
    times: Tensor,
    *,
    target_states: Tensor | None = None,
    nominal_controls: Tensor | None = None,
    method: str = "rk4",
) -> tuple[Tensor, Tensor]:
    """Roll out dynamics using an LQR-style feedback controller."""
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if times.numel() < 2:
        raise ValueError("at least two time samples are required")

    state = initial_state
    states = [state]
    controls = []
    for step in range(times.numel() - 1):
        target = None if target_states is None else target_states[step]
        nominal = None if nominal_controls is None else nominal_controls[step]
        control = controller.control(
            state,
            step,
            target_state=target,
            nominal_control=nominal,
        )
        controls.append(control)
        dt = (times[step + 1] - times[step]).to(dtype=state.dtype, device=state.device)
        state = discrete_step(
            dynamics,
            times[step].to(dtype=state.dtype, device=state.device),
            state,
            dt,
            control,
            method=method,
        )
        states.append(state)

    return torch.stack(states, dim=0), torch.stack(controls, dim=0)
