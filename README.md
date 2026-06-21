# Mechanica

Differentiable analytical and classical mechanics in PyTorch.

Mechanica provides small (for now), composable tools for Lagrangian systems,
Hamiltonian systems, Newtonian mechanics, trajectory diagnostics, and inverse
parameter fitting.

## Install

```bash
pip install mechanica-torch
```

For local development:

```bash
pip install -e ".[dev]"
```

## Optional native kernels

Mechanica is pure PyTorch by default. The spring force helper also has an
optional C++/Torch extension path for local experiments:

```bash
pip install -e ".[native]"
```

```python
forces = hooke_spring_force(
    positions,
    edges,
    rest_lengths=1.0,
    stiffness=10.0,
    use_native=True,
)
```

You can also set `MECHANICA_USE_NATIVE=1` to request the native path globally.
If the extension cannot compile on that machine, env-based native loading falls
back to the pure Torch implementation with a warning. Native loading requires
PyTorch's extension build toolchain (`ninja` plus a local C++ compiler).

## Example

```python
import torch
from mechanica import LagrangianSystem

m = torch.tensor(1.0)
k = torch.tensor(4.0)

def kinetic(q, qdot):
    return 0.5 * m * (qdot * qdot).sum()

def potential(q):
    return 0.5 * k * (q * q).sum()

system = LagrangianSystem(kinetic=kinetic, potential=potential)

t = torch.linspace(0, 2, 64)
w = torch.sqrt(k / m)
q = torch.cos(w * t).unsqueeze(-1)
qdot = -w * torch.sin(w * t).unsqueeze(-1)
qddot = -(w**2) * torch.cos(w * t).unsqueeze(-1)

residual = system.euler_lagrange_residual(q, qdot, qddot)
print(residual.abs().max())
```

## What we're focusing on right now

- Euler-Lagrange and Hamilton residuals
- generalized momenta and Noether-style charges
- Poisson brackets and canonical transformation checks
- energy, momentum, springs, gravity, and Newton residuals
- differentiable integrators
- inverse mechanics from trajectory data
- state-space dynamics for control, robotics, and learned systems
- finite-horizon LQR and TVLQR around differentiable dynamics
- Lagrangian manipulator dynamics for inverse/forward dynamics

Mechanica is tensor-first, not symbolic. It uses PyTorch autograd to inspect
concrete systems and trajectories.

## Dynamics, control, and robotics

Mechanica can translate mechanics models into the state-space form used by
control and robotics:

```python
from mechanica import finite_horizon_lqr, hamiltonian_dynamics, rollout

dynamics = hamiltonian_dynamics(system)  # x = [q, p], u adds generalized force
trajectory = rollout(dynamics, initial_state, times, controls=controls)
lqr = finite_horizon_lqr(A, B, Q, R, horizon=50)
u = lqr.control(state, step=0)
```

For Lagrangian robotics models:

```python
from mechanica import forward_dynamics, inverse_dynamics, lagrangian_state_dynamics

tau = inverse_dynamics(system, q, qdot, qddot_desired)
qddot = forward_dynamics(system, q, qdot, tau)
dynamics = lagrangian_state_dynamics(system)  # x = [q, qdot], u = tau
```

## Learning mechanics

Torch modules can be wrapped as learned energies and trained with residual or
rollout losses:

```python
from mechanica import LagrangianModule, fit_rollout, lagrangian_residual_loss

model = LagrangianModule(kinetic=kinetic_fn, potential=potential_net)
loss = lagrangian_residual_loss(model.system(), q, qdot, qddot)
result = fit_rollout(dynamics, x0, times, observed_states, model.parameters())
```

## Analytical structure

```python
import torch
from mechanica import HamiltonianSystem, LagrangianSystem, canonical_transformation_residual

m = torch.tensor(1.0)
k = torch.tensor(4.0)

def kinetic(q, qdot):
    return 0.5 * m * (qdot * qdot).sum()

def potential(q):
    return 0.5 * k * (q * q).sum()

lagrangian = LagrangianSystem(kinetic=kinetic, potential=potential)
q = torch.tensor([[1.0, 0.0]])
qdot = torch.tensor([[0.0, 2.0]])

def rotation(q):
    return torch.stack([-q[1], q[0]])

angular_charge = lagrangian.noether_charge(q, qdot, rotation)
symmetry_error = lagrangian.coordinate_symmetry_residual(q, qdot, rotation)

def hamiltonian(q, p):
    return 0.5 * (p * p).sum() / m + 0.5 * k * (q * q).sum()

system = HamiltonianSystem(hamiltonian)

def position(q, p):
    return q[0]

def momentum(q, p):
    return p[0]

canonical_relation = system.poisson_bracket(position, momentum, q[:, :1], qdot[:, :1])

def phase_rotation(q, p):
    return p, -q

canonical_error = canonical_transformation_residual(phase_rotation, q, qdot)
```
