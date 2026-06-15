# Lid-Driven Cavity Flow

This section contains the numerical implementation of the two-dimensional lid-driven cavity benchmark problem.

The case consists of an incompressible flow inside a square cavity, where the top wall moves with a prescribed horizontal velocity while the remaining walls are stationary. This benchmark is widely used to validate incompressible Navier-Stokes solvers.

## Implemented versions

| Folder         | Description                                                            |
| -------------- | ---------------------------------------------------------------------- |
| `uniform_mesh` | Solver using a uniform Cartesian MAC grid                              |
| `general_mesh` | Solver using a non-uniform MAC grid with optional tanh mesh refinement |

## Numerical method

The solver is based on:

* Finite volume discretization
* Staggered MAC grid arrangement
* Fractional-step projection method
* Explicit AB2 treatment of the momentum equations
* Central or upwind discretization of convective terms
* Dense pressure Poisson matrix inversion
* Velocity correction to enforce incompressibility

## Physical outputs

The simulations compute and store:

* Horizontal velocity field
* Vertical velocity field
* Pressure field
* Velocity magnitude
* Vorticity field
* Streamfunction field
* Divergence diagnostics
* Centerline velocity profiles

## Validation

The lid-driven cavity solution can be compared against reference benchmark data, particularly the velocity profiles reported by Ghia et al. for selected Reynolds numbers.

## Notes

This implementation is intended for educational and validation purposes. The dense pressure-Poisson formulation is useful for clarity and reproducibility, but sparse matrices and iterative solvers would be more appropriate for larger-scale simulations.
