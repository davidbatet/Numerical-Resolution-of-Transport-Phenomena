# Differentially Heated Cavity

This section contains the numerical implementation of the two-dimensional differentially heated cavity problem.

The case represents natural convection in a square cavity with a hot vertical wall, a cold vertical wall and adiabatic horizontal walls. The flow is driven by buoyancy forces and is modeled using the incompressible Navier-Stokes equations coupled with the temperature transport equation under the Boussinesq approximation.

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
* Explicit AB2 treatment of the temperature equation
* Boussinesq approximation for buoyancy coupling
* Dense pressure Poisson matrix inversion
* Nusselt-number evaluation at the hot and cold walls

## Physical outputs

The simulations compute and store:

* Velocity field
* Pressure field
* Temperature field
* Vorticity field
* Nusselt number
* Streamfunction field
* Divergence diagnostics
* de Vahl Davis benchmark quantities

## Notes

This implementation is intended for educational and validation purposes. The dense pressure-Poisson formulation is useful for clarity and reproducibility, but sparse matrices and iterative solvers would be more appropriate for larger-scale simulations.
