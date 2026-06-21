import torch

from mechanica import (
    LagrangianModule,
    fit_rollout,
    lagrangian_residual_loss,
    rollout,
    second_order_dynamics,
)


def test_lagrangian_module_residual_loss_backpropagates_to_energy_parameters() -> None:
    mass = torch.tensor(1.0)

    def kinetic(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        return 0.5 * mass * (qdot * qdot).sum()

    class Potential(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.log_k = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, q: torch.Tensor) -> torch.Tensor:
            return 0.5 * torch.exp(self.log_k) * (q * q).sum()

    model = LagrangianModule(kinetic=kinetic, potential=Potential())
    system = model.system()

    true_k = torch.tensor(4.0)
    times = torch.linspace(0.0, 1.0, 16)
    omega = torch.sqrt(true_k / mass)
    q = torch.cos(omega * times).unsqueeze(-1)
    qdot = -omega * torch.sin(omega * times).unsqueeze(-1)
    qddot = -(omega**2) * torch.cos(omega * times).unsqueeze(-1)

    loss = lagrangian_residual_loss(system, q, qdot, qddot)
    loss.backward()

    assert model.potential.log_k.grad is not None  # type: ignore[union-attr]


def test_fit_rollout_recovers_constant_acceleration() -> None:
    learned_acceleration = torch.nn.Parameter(torch.tensor(0.0))
    true_acceleration = torch.tensor(2.0)

    def learned_dynamics(
        time: torch.Tensor,
        state: torch.Tensor,
        control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del time, control
        return torch.stack([state[1], learned_acceleration])

    def true_acceleration_fn(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        del qdot
        return true_acceleration.expand_as(q)

    times = torch.linspace(0.0, 1.0, 24)
    initial_state = torch.tensor([0.0, 0.0])
    target = rollout(second_order_dynamics(true_acceleration_fn), initial_state, times)

    result = fit_rollout(
        learned_dynamics,
        initial_state,
        times,
        target,
        [learned_acceleration],
        steps=120,
        lr=0.1,
    )

    assert result.final_loss < 1e-4
    assert torch.allclose(learned_acceleration.detach(), true_acceleration, atol=1e-2)
