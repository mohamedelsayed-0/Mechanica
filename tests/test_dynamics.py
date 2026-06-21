import torch

from mechanica import (
    HamiltonianSystem,
    hamiltonian_dynamics,
    join_state,
    linearize,
    rollout,
    second_order_dynamics,
    split_state,
)


def test_join_and_split_state_round_trip() -> None:
    q = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    v = torch.tensor([[0.5, -0.5], [1.5, -1.5]])

    state = join_state(q, v)
    next_q, next_v = split_state(state)

    assert torch.allclose(next_q, q)
    assert torch.allclose(next_v, v)


def test_second_order_rollout_matches_harmonic_oscillator() -> None:
    def acceleration(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        return -q

    dynamics = second_order_dynamics(acceleration)
    times = torch.linspace(0.0, 1.0, 64)
    trajectory = rollout(dynamics, torch.tensor([1.0, 0.0]), times)

    expected = torch.tensor([torch.cos(times[-1]), -torch.sin(times[-1])])
    assert torch.allclose(trajectory[-1], expected, atol=1e-5)


def test_linearize_controlled_double_integrator() -> None:
    def acceleration(
        q: torch.Tensor,
        qdot: torch.Tensor,
        control: torch.Tensor | None,
    ) -> torch.Tensor:
        assert control is not None
        return control

    dynamics = second_order_dynamics(acceleration)
    state = torch.tensor([2.0, -1.0])
    control = torch.tensor([3.0])

    a_matrix, b_matrix = linearize(dynamics, 0.0, state, control)

    assert torch.allclose(a_matrix, torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
    assert torch.allclose(b_matrix, torch.tensor([[0.0], [1.0]]))


def test_hamiltonian_dynamics_adapts_to_state_space_with_control() -> None:
    mass = torch.tensor(2.0)
    stiffness = torch.tensor(8.0)

    def hamiltonian(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return 0.5 * (p * p).sum() / mass + 0.5 * stiffness * (q * q).sum()

    dynamics = hamiltonian_dynamics(HamiltonianSystem(hamiltonian))
    derivative = dynamics(torch.tensor(0.0), torch.tensor([0.5, 2.0]), torch.tensor([1.0]))

    assert torch.allclose(derivative, torch.tensor([1.0, -3.0]))
