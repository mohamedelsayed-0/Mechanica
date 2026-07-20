# Focus

This library is centered on analytical mechanics, with classical mechanics as the
practical layer around it.

## Core Identity

Mechanica should make mechanics differentiable and inspectable:

- define a Lagrangian or Hamiltonian
- evaluate equation residuals on a trajectory
- fit unknown physical parameters
- measure where observed trajectories depart from a modeled system

## What Belongs

- Lagrangian mechanics
- Hamiltonian mechanics
- Newtonian particle mechanics
- conservation laws
- generalized momenta and coordinate symmetry residuals
- Poisson brackets and observable time derivatives
- canonical transformation residuals for phase-space maps
- trajectory fitting
- physics residuals as Torch losses
- differentiable integrators
- differentiable 3D robot kinematics and rigid-body dynamics
- robot constraints, primitive collision distances, and trajectory optimization
- data adapters that produce trajectories for the mechanics core

## What Does Not Belong Yet

- full CFD solvers
- full finite-element engines
- quantum mechanics
- symbolic algebra systems
- automatic symbolic symmetry discovery
- full field-theory variation engines
- Hamilton-Jacobi solvers before the tensor mechanics core is stable
- engine-specific wrappers before the core API is stable

## First Real Use Case

Given observed coordinates, velocities, and accelerations, report:

- Euler-Lagrange residual
- energy drift
- momentum drift
- coordinate symmetry residual
- Poisson bracket checks for conserved observables
- best-fit physical parameters
- sample locations where the assumed mechanics breaks
