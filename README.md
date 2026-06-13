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

Mechanica is tensor-first, not symbolic. It uses PyTorch autograd to inspect
concrete systems and trajectories.

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
