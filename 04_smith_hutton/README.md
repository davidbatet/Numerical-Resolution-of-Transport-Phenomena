# Smith-Hutton Benchmark

This section contains the numerical implementation of the Smith-Hutton convection-diffusion benchmark problem.

The Smith-Hutton problem is a steady two-dimensional scalar transport case with a prescribed velocity field. It is commonly used to assess the numerical behavior of convection-diffusion discretization schemes, especially in convection-dominated regimes.

## Implemented versions

| Folder         | Description                                                      |
| -------------- | ---------------------------------------------------------------- |
| `uniform_mesh` | Solver using a uniform Cartesian mesh                            |
| `refined_mesh` | Solver using a general mesh formulation with optional refinement |

## Numerical method

The solver is based on:

* Finite volume discretization
* Structured Cartesian mesh
* Prescribed incompressible velocity field
* Steady convection-diffusion equation
* Upwind and power-law discretization schemes
* Patankar-style coefficient formulation
* Dense matrix inversion
* Pseudo-transient iterative convergence

## Physical outputs

The simulations compute and store:

* Scalar field
* Outlet scalar profile
* Convergence history
* Comparison data at the outlet
* VTK output for visualization

## Validation

The solution can be validated by comparing the scalar distribution along the outlet boundary against reference Smith-Hutton benchmark data for different values of the diffusion parameter.

## Notes

This implementation is intended for educational and validation purposes. The benchmark is especially useful for evaluating numerical diffusion and the influence of the convection-diffusion discretization scheme.
