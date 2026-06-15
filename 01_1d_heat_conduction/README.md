# 1D Heat Conduction

This section contains the numerical implementation of one-dimensional heat conduction problems.

The cases include steady and transient formulations, internal heat generation and different thermal boundary conditions. These examples represent the first validation stage of the finite volume framework before extending the methodology to two-dimensional transport problems.

## Implemented versions

| Folder      | Description                          |
| ----------- | ------------------------------------ |
| `steady`    | Steady 1D heat conduction solvers    |
| `transient` | Transient 1D heat conduction solvers |

## Numerical method

The solvers are based on:

* Finite volume discretization
* One-dimensional structured mesh
* Steady and transient heat conduction equations
* Internal heat generation
* Dirichlet boundary conditions
* Robin convective boundary conditions
* Explicit and implicit time integration
* Analytical solution comparison

## Physical outputs

The simulations compute and store:

* Temperature distribution
* Transient temperature evolution
* Analytical reference solution
* Numerical error with respect to the analytical solution

## Notes

This implementation is intended as a first verification step for the finite volume methodology. The one-dimensional cases are useful for validating coefficient assembly, boundary-condition implementation and transient time integration before moving to two-dimensional conduction and convection-diffusion problems.
