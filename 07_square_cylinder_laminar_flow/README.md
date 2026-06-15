# Square Cylinder Laminar Flow

This section contains the numerical implementation of the two-dimensional laminar flow around a square cylinder in a channel.

The problem is solved using the incompressible Navier-Stokes equations on a staggered MAC grid. The square cylinder is represented using an immersed boundary method based on Peskin-type interpolation and spreading functions.

## Implemented versions

| Folder         | Description                                                                      |
| -------------- | -------------------------------------------------------------------------------- |
| `uniform_mesh` | Solver using a uniform Cartesian MAC grid                                        |
| `refined_mesh` | Solver using a non-uniform refined MAC grid clustered around the square cylinder |

## Numerical method

The solver is based on:

* Finite volume discretization
* Staggered MAC grid arrangement
* Fractional-step projection method
* Explicit AB2 treatment of the momentum equations
* Central or upwind discretization of the convective terms
* Dense pressure Poisson matrix inversion
* Immersed boundary forcing for the square cylinder
* Drag and lift coefficient evaluation
* Strouhal-number estimation from the lift or velocity signal

## Physical outputs

The simulations compute and store:

* Velocity field
* Pressure field
* Vorticity field
* Drag coefficient
* Lift coefficient
* Probe velocity signal
* Divergence diagnostics
* Strouhal number

## Notes

This implementation is intended for educational and validation purposes. The dense pressure-Poisson formulation is useful for clarity and reproducibility, but sparse matrices and iterative solvers would be more appropriate for larger-scale simulations.
