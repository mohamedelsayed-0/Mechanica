# Mechanica

Differentiable mechanics, control, and robotics tools for PyTorch.

Mechanica is a tensor-first library for translating physical systems into
residuals, dynamics, controllers, and robotics computations. It uses Torch
autograd instead of symbolic algebra, so learned models and classical mechanics
can share the same workflow.

## Install

```bash
pip install mechanica-torch
```

For development:

```bash
pip install -e ".[dev]"
```

Optional native kernels need PyTorch's extension toolchain:

```bash
pip install -e ".[native]"
```

## Quick Example

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
qddot = -(w**2) * q

residual = system.euler_lagrange_residual(q, qdot, qddot)
print(residual.abs().max())
```

## What It Covers

- Lagrangian and Hamiltonian residuals
- finite differences, diagnostics, and inverse fitting
- state-space rollout, linearization, LQR, TVLQR, and feedback rollout
- manipulator dynamics: mass matrices, bias/gravity terms, inverse/forward dynamics
- planar serial-chain kinematics and Jacobians
- classical particle mechanics and spring forces
- optional C++/Torch spring kernels
- learned Lagrangian and Hamiltonian modules

## Examples

```bash
python examples/pendulum_tvlqr.py
python examples/two_link_kinematics.py
python examples/native_spring_benchmark.py
```

## Native Kernels

Mechanica is pure PyTorch by default. Set `use_native=True` on supported
helpers, or `MECHANICA_USE_NATIVE=1`, to try optional C++/Torch kernels. Use
`native_spring_status()` to check whether the local extension toolchain is
available. Set `MECHANICA_NATIVE_BUILD_DIR` to choose the build cache location.
