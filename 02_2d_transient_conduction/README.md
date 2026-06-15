# 2D Transient Heat Conduction

This section contains the numerical implementation of a two-dimensional transient heat conduction problem.

The case solves the unsteady temperature field in a solid domain using a finite volume formulation. The implementation supports heterogeneous thermal properties and mixed thermal boundary conditions.

## Numerical method

The solver is based on:

* Finite volume discretization
* Structured Cartesian mesh
* Transient heat conduction equation
* Implicit time integration
* Heterogeneous material properties
* Mixed thermal boundary conditions
* Dense matrix assembly and inversion
* Time-dependent temperature-field evolution

## Physical outputs

The simulations compute and store:

* Temperature field
* Time evolution of the solution
* Material-property distribution
* Boundary-condition effects
* VTK output for visualization

## Notes

This implementation is intended for educational and validation purposes. The dense matrix formulation is useful for clarity and for understanding the finite volume discretization process, although sparse matrices and iterative linear solvers would be more appropriate for larger domains.
