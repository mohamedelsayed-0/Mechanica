import torch

from mechanica.trajectory import direct_collocation_residual, ilqr


def discrete_double_integrator(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    dt = 0.1
    return torch.stack((state[0] + dt * state[1], state[1] + dt * control[0]))


def stage_cost(state: torch.Tensor, control: torch.Tensor, step: int) -> torch.Tensor:
    del step
    return state.square().sum() + 0.01 * control.square().sum()


def terminal_cost(state: torch.Tensor) -> torch.Tensor:
    return 20 * state.square().sum()


def test_ilqr_and_ddp_reduce_control_problem_cost() -> None:
    initial_state = torch.tensor([1.0, 0.0], dtype=torch.float64)
    controls = torch.zeros(20, 1, dtype=torch.float64)
    for method in ("ilqr", "ddp"):
        result = ilqr(
            discrete_double_integrator,
            initial_state,
            controls,
            stage_cost,
            terminal_cost,
            method=method,
            max_iterations=10,
        )
        assert result.cost_history[-1] < result.cost_history[0]
        assert result.states[-1].norm() < initial_state.norm()


def test_direct_collocation_is_zero_for_constant_velocity() -> None:
    times = torch.tensor([0.0, 0.5, 1.0])
    states = torch.tensor([[0.0], [0.5], [1.0]])
    controls = torch.zeros(2, 1)
    residual = direct_collocation_residual(
        lambda time, state, control: torch.ones_like(state), states, controls, times
    )
    assert torch.allclose(residual, torch.zeros_like(residual))


def test_batched_ilqr_and_control_limits() -> None:
    states = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    controls = torch.zeros(2, 10, 1, dtype=torch.float64)
    result = ilqr(
        discrete_double_integrator,
        states,
        controls,
        stage_cost,
        terminal_cost,
        max_iterations=5,
        control_limits=(torch.tensor([-0.5]), torch.tensor([0.5])),
    )
    assert result.states.shape == (2, 11, 2)
    assert result.controls.abs().max() <= 0.5
    assert torch.all(result.cost_history[:, -1] < result.cost_history[:, 0])
