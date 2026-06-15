# Finite Volume CFD and Heat-Transfer Solvers

This repository contains a collection of self-developed numerical solvers for heat transfer, convection-diffusion and incompressible fluid-flow benchmark problems.

The solvers were developed as part of the Bachelor's Thesis:

**Numerical implementation of CFD and heat-transfer solvers with self-developed applications**

The main objective of the project is to implement, from scratch, a progressive numerical framework for the resolution of transport phenomena using Python and NumPy.

## Implemented cases

| Folder                            | Problem                      | Main numerical features                                                       |
| --------------------------------- | ---------------------------- | ----------------------------------------------------------------------------- |
| `01_1d_heat_conduction`           | 1D heat conduction           | Finite volume method, steady and transient formulations                       |
| `02_2d_transient_conduction`      | 2D transient conduction      | Implicit finite volume formulation, heterogeneous material properties         |
| `03_2d_convection_diffusion`      | 2D convection-diffusion      | Ghost-cell and Patankar-style boundary treatments                             |
| `04_smith_hutton`                 | Smith-Hutton benchmark       | Steady convection-diffusion validation case                                   |
| `05_lid_driven_cavity`            | Lid-driven cavity flow       | MAC grid, projection method, benchmark validation                             |
| `06_differentially_heated_cavity` | Differentially heated cavity | Natural convection, Boussinesq approximation, Nusselt-number evaluation       |
| `07_square_cylinder_laminar_flow` | Square cylinder laminar flow | MAC grid, immersed boundary method, drag, lift and Strouhal-number estimation |

## Numerical methods

The numerical solvers are based on finite volume discretizations on structured Cartesian grids. Heat-transfer and scalar transport problems are solved using steady, transient, explicit, implicit and pseudo-transient formulations depending on the benchmark case.

The incompressible Navier-Stokes equations are solved using a fractional-step projection method on staggered MAC grids. Convective terms are treated explicitly, using central or upwind discretizations depending on the case. Pressure Poisson equations are assembled as dense matrices for educational and validation purposes.

Some advanced cases include non-uniform mesh refinement, Boussinesq coupling for natural convection and immersed-boundary forcing for the square-cylinder flow problem.

## Repository structure

```text
finite-volume-cfd-heat-transfer/
├── 01_1d_heat_conduction/
├── 02_2d_transient_conduction/
├── 03_2d_convection_diffusion/
├── 04_smith_hutton/
├── 05_lid_driven_cavity/
├── 06_differentially_heated_cavity/
├── 07_square_cylinder_laminar_flow/
├── LICENSE
├── README.md
└── requirements.txt
```

## Requirements

The code requires Python 3 and the following packages:

```text
numpy
matplotlib
numba
```

They can be installed with:

```bash
pip install -r requirements.txt
```

## Notes

The objective of this repository is educational and methodological. The solvers are intentionally implemented from first principles in order to expose the numerical formulation, discretization strategy and validation process.

For large-scale simulations, sparse matrices, iterative linear solvers and optimized data structures would be more appropriate than the dense-matrix approach used in several examples.
