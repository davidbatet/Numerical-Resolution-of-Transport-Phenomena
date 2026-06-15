# 2D Convection-Diffusion

This section contains the numerical implementation of a two-dimensional steady convection-diffusion problem.

The case solves the transport of a scalar quantity in a rectangular domain with prescribed velocity components, diffusion coefficient and boundary conditions. Two different boundary-condition treatments are implemented in order to compare their numerical behavior.

## Implemented versions

| Folder              | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| `ghost_cells`       | Solver using ghost-cell boundary treatment                 |
| `patankar_boundary` | Solver using Patankar-style boundary coefficient treatment |

## Numerical method

The solver is based on:

* Finite volume discretization
* Structured Cartesian mesh
* Steady convection-diffusion equation
* Prescribed velocity field
* Upwind and power-law discretization schemes
* Ghost-cell boundary treatment
* Patankar-style boundary coefficient formulation
* Dense matrix inversion
* Pseudo-transient iterative convergence

## Physical outputs

The simulations compute and store:

* Scalar field
* Convergence history
* Boundary-condition comparison
* VTK output for visualization

## Notes

This implementation is intended for educational and validation purposes. The case is useful for comparing different finite volume treatments of convection-diffusion boundary conditions and for assessing the influence of numerical diffusion.
