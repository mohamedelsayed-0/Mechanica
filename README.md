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

Build once in place to avoid runtime compilation:

```bash
python setup_native.py build_ext --inplace
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
- URDF robot trees, batched SE(3) kinematics, RNEA, CRBA, and ABA
- inverse kinematics, operational-space control, constraints, and collision costs
- batched iLQR, DDP, MPC helpers, and direct-collocation defects
- classical particle mechanics, springs, and pairwise gravity
- batched spring graphs and cutoff gravity neighbor lists
- holonomic constraints, projection, and RATTLE integration
- optional prebuilt or JIT-compiled C++/Torch kernels
- learned Lagrangian and Hamiltonian modules

## Examples

```bash
python examples/pendulum_tvlqr.py
python examples/two_link_kinematics.py
python examples/native_spring_benchmark.py
python examples/urdf_robotics.py
```

## Differentiable Robots

```python
import torch
from mechanica import forward_kinematics, load_urdf, mass_matrix_crba

model = load_urdf("robot.urdf").to(dtype=torch.float32)
q = torch.zeros(model.dof, requires_grad=True)
poses = forward_kinematics(model, q)
mass = mass_matrix_crba(model, q)
(poses[-1, :3, 3].square().sum() + mass.trace()).backward()
```

The optional native extension registers batched `SO(3)` and robot-tree forward
kinematics as Torch operators, preserving autograd and device dispatch.

## Native Kernels

Mechanica is pure PyTorch by default. Set `use_native=True` on supported
helpers, or `MECHANICA_USE_NATIVE=1`, to try optional C++/Torch kernels. Use
`native_kernels_status()` to check whether the local extension toolchain is
available. Set `MECHANICA_NATIVE_BUILD_DIR` to choose the build cache location.
Prebuilt extensions are preferred automatically when present.
