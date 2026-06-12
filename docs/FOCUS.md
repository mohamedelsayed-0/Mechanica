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
- trajectory fitting
- physics residuals as Torch losses
- differentiable integrators
- data adapters that produce trajectories for the mechanics core

## What Does Not Belong Yet

- full CFD solvers
- full finite-element engines
- quantum mechanics
- symbolic algebra systems
- engine-specific wrappers before the core API is stable

## First Real Use Case

Given observed coordinates, velocities, and accelerations, report:

- Euler-Lagrange residual
- energy drift
- momentum drift
- best-fit physical parameters
- sample locations where the assumed mechanics breaks
