"""Stabilize an upright pendulum with Lagrangian dynamics and TVLQR."""

import torch

from mechanica import LagrangianSystem, feedback_rollout, lagrangian_state_dynamics, tvlqr

mass = torch.tensor(1.0)
length = torch.tensor(1.0)
gravity = torch.tensor(9.80665)
inertia = mass * length * length


def kinetic(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
    return 0.5 * inertia * (qdot * qdot).sum()


def potential(q: torch.Tensor) -> torch.Tensor:
    return mass * gravity * length * (1 - torch.cos(q[0]))


system = LagrangianSystem(kinetic=kinetic, potential=potential, name="pendulum")
dynamics = lagrangian_state_dynamics(system)

times = torch.linspace(0.0, 4.0, 81)
upright = torch.tensor([torch.pi, 0.0])
nominal_states = upright.repeat(times.numel(), 1)
nominal_controls = torch.zeros(times.numel() - 1, 1)

controller = tvlqr(
    dynamics,
    times,
    nominal_states,
    nominal_controls,
    Q=torch.diag(torch.tensor([20.0, 2.0])),
    R=torch.tensor([[0.1]]),
    Qf=torch.diag(torch.tensor([60.0, 6.0])),
)

trajectory, torques = feedback_rollout(
    dynamics,
    controller,
    torch.tensor([torch.pi + 0.25, 0.0]),
    times,
)

initial_error = (trajectory[0, 0] - torch.pi).abs()
final_error = (trajectory[-1, 0] - torch.pi).abs()
print("initial angle error:", float(initial_error.detach()))
print("final angle error:", float(final_error.detach()))
print("max torque:", float(torques.abs().max().detach()))
