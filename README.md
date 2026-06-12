# Mechanica

Differentiable analytical and classical mechanics in PyTorch.

Mechanica provides small (for now), composable tools for Lagrangian systems, Hamiltonian
systems, Newtonian mechanics, trajectory diagnostics, and inverse parameter
fitting.

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

- Euler-Lagrange and Hamilton equation residuals
- energy, momentum, springs, gravity, and Newton residuals
- differentiable integrators
- inverse mechanics from trajectory data
