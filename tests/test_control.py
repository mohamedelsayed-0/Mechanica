import torch

from mechanica import (
    feedback_rollout,
    finite_horizon_lqr,
    infinite_horizon_lqr,
    quadratic_cost,
    rollout_cost,
    second_order_dynamics,
    tvlqr,
)


def test_finite_horizon_lqr_stabilizes_double_integrator_direction() -> None:
    dt = 0.1
    a_matrix = torch.tensor([[1.0, dt], [0.0, 1.0]])
    b_matrix = torch.tensor([[0.5 * dt * dt], [dt]])
    q_matrix = torch.diag(torch.tensor([1.0, 0.1]))
    r_matrix = torch.tensor([[0.01]])

    result = finite_horizon_lqr(a_matrix, b_matrix, q_matrix, r_matrix, horizon=30)
    control = result.control(torch.tensor([1.0, 0.0]), 0)

    assert result.gains.shape == (30, 1, 2)
    assert result.value_matrices.shape == (31, 2, 2)
    assert control.shape == (1,)
    assert control[0] < 0


def test_infinite_horizon_lqr_stabilizes_discrete_double_integrator() -> None:
    dt = 0.1
    a_matrix = torch.tensor([[1.0, dt], [0.0, 1.0]])
    b_matrix = torch.tensor([[0.5 * dt * dt], [dt]])
    q_matrix = torch.diag(torch.tensor([1.0, 0.1]))
    r_matrix = torch.tensor([[0.01]])

    result = infinite_horizon_lqr(a_matrix, b_matrix, q_matrix, r_matrix)
    closed_loop = a_matrix - b_matrix @ result.gains[0]

    assert result.converged
    assert result.gains.shape == (1, 1, 2)
    assert torch.linalg.eigvals(closed_loop).abs().max() < 1


def test_tvlqr_linearizes_state_space_dynamics() -> None:
    def acceleration(
        q: torch.Tensor,
        qdot: torch.Tensor,
        control: torch.Tensor | None,
    ) -> torch.Tensor:
        assert control is not None
        return control

    dynamics = second_order_dynamics(acceleration)
    times = torch.linspace(0.0, 0.3, 4)
    states = torch.zeros(4, 2)
    controls = torch.zeros(3, 1)
    q_matrix = torch.eye(2)
    r_matrix = torch.eye(1)

    result = tvlqr(dynamics, times, states, controls, q_matrix, r_matrix)

    assert result.gains.shape == (3, 1, 2)
    assert torch.allclose(result.discrete_A[0], torch.tensor([[1.0, 0.1], [0.0, 1.0]]))
    assert torch.allclose(result.discrete_B[0], torch.tensor([[0.0], [0.1]]))


def test_quadratic_cost_is_differentiable() -> None:
    states = torch.tensor([[1.0, 0.0], [0.5, 0.1]], requires_grad=True)
    controls = torch.tensor([[0.2]], requires_grad=True)

    cost = quadratic_cost(states, controls, torch.eye(2), torch.eye(1))
    cost.backward()

    assert cost > 0
    assert states.grad is not None
    assert controls.grad is not None


def test_rollout_cost_composes_dynamics_and_quadratic_cost() -> None:
    def acceleration(
        q: torch.Tensor,
        qdot: torch.Tensor,
        control: torch.Tensor | None,
    ) -> torch.Tensor:
        assert control is not None
        return control

    dynamics = second_order_dynamics(acceleration)
    times = torch.linspace(0.0, 0.2, 3)
    controls = torch.zeros(2, 1, requires_grad=True)
    cost = rollout_cost(
        dynamics,
        torch.tensor([1.0, 0.0]),
        times,
        controls,
        torch.eye(2),
        torch.eye(1),
    )
    cost.backward()

    assert cost > 0
    assert controls.grad is not None


def test_feedback_rollout_applies_lqr_controller() -> None:
    def acceleration(
        q: torch.Tensor,
        qdot: torch.Tensor,
        control: torch.Tensor | None,
    ) -> torch.Tensor:
        assert control is not None
        return control

    dt = 0.1
    a_matrix = torch.tensor([[1.0, dt], [0.0, 1.0]])
    b_matrix = torch.tensor([[0.5 * dt * dt], [dt]])
    controller = infinite_horizon_lqr(
        a_matrix,
        b_matrix,
        torch.diag(torch.tensor([1.0, 0.1])),
        torch.tensor([[0.01]]),
    )
    states, controls = feedback_rollout(
        second_order_dynamics(acceleration),
        controller,
        torch.tensor([1.0, 0.0]),
        torch.linspace(0.0, 2.0, 21),
    )

    assert controls.shape == (20, 1)
    assert states[-1].norm() < states[0].norm()
