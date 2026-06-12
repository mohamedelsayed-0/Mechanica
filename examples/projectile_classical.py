"""Classical mechanics diagnostics for a projectile trajectory."""

import torch

from mechanica import estimate_acceleration, estimate_velocity, newton_residual

dt = 0.01
t = torch.arange(0, 1, dt)
g = 9.80665
m = torch.tensor(2.0)

initial_position = torch.tensor([0.0, 1.0])
initial_velocity = torch.tensor([3.0, 6.0])

positions = initial_position + t[:, None] * initial_velocity
positions[:, 1] -= 0.5 * g * t * t

velocities = estimate_velocity(positions, dt)
accelerations = estimate_acceleration(positions, dt)

forces = torch.zeros_like(accelerations)
forces[:, 1] = -m * g

residual = newton_residual(accelerations, forces, m)
print("mean Newton residual:", residual.norm(dim=-1).mean().item())
