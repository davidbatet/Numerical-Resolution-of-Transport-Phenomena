"""
2D transient convection-diffusion problem with split inlet boundary condition.

Numerical method:
    - Cell-centered finite volume method
    - Ghost-cell treatment for boundary conditions
    - Fully implicit transient formulation
    - Uniform structured Cartesian mesh
    - Upwind or power-law convection-diffusion scheme
    - Dense global matrix assembly
    - Precomputed inverse matrix reused during the transient loop

Physical problem:
    dphi/dt + div(u phi) = div(Gamma grad(phi))

Boundary conditions:
    - Left boundary: split Dirichlet inlet
    - Right boundary: zero-gradient outlet
    - Top boundary: adiabatic / zero-gradient
    - Bottom boundary: adiabatic / zero-gradient

Author: David Batet Romero
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from functions import (
    build_mesh_ghost,
    fill_ghost_phi,
    assemble_matrix_and_source_ghost,
    invert_matrix,
    solve_scalar_field,
    write_paraview_structured_grid_vtk,
)


# Problem geometry
Lx = 1.10
Ly = 0.80

Nx = 110
Ny = 80

y_split = Ly / 2.0


# Diffusion coefficient
Gamma = {
    "TOP": 2.0e-3,
    "BOT": 2.0e-3,
}


# Velocity field
u = 0.01
v = 0.0


# Boundary values
phi_in_top = 40.0
phi_in_bottom = 0.0


# Initial and time parameters
phi_initial = 0.0

t_final = 5000.0
dt = 100.0
nsteps = int(t_final / dt)

tolerance = 1.0e-3


# Numerical scheme
scheme = "powerlaw"  # Available options: "upwind", "powerlaw"


# Output directory
base_dir = os.path.dirname(os.path.abspath(__file__))
vtk_dir = os.path.join(base_dir, "vtk")
os.makedirs(vtk_dir, exist_ok=True)


# Mesh generation
mesh = build_mesh_ghost(Lx, Ly, Nx, Ny)

x_c = mesh["x_c"]
y_c = mesh["y_c"]

dx_cell = mesh["dx_cell"]
dy_cell = mesh["dy_cell"]

Xc = mesh["Xc"]
Yc = mesh["Yc"]


# Cell-wise diffusion coefficient
Gamma_cell = np.where(
    Yc >= y_split,
    Gamma["TOP"],
    Gamma["BOT"],
).astype(float)


# Transient coefficient
cell_area = dy_cell[:, None] * dx_cell[None, :]
aP0 = cell_area / dt


# Scalar field including ghost cells
phi_g = np.full((Ny + 2, Nx + 2), phi_initial, dtype=float)


# Initial ghost-cell update
fill_ghost_phi(
    phi_g,
    x_c,
    y_c,
    Lx,
    Ly,
    phi_in_top,
    phi_in_bottom,
    y_split,
    top_adiabatic=True,
    bottom_adiabatic=True,
    outlet_zero_grad=True,
)


# Matrix and source assembly
K, Su = assemble_matrix_and_source_ghost(
    Nx,
    Ny,
    x_c,
    y_c,
    dx_cell,
    dy_cell,
    Gamma_cell,
    u,
    v,
    aP0,
    phi_in_top,
    phi_in_bottom,
    y_split,
    scheme=scheme,
    top_adiabatic=True,
    bottom_adiabatic=True,
    outlet_zero_grad=True,
)


# Matrix inversion
print("Inverting global matrix...", flush=True)
K_inv = invert_matrix(K)
print("Starting time integration...", flush=True)


# Live plot
plt.ion()

fig, ax = plt.subplots(figsize=(7, 4))

im = ax.imshow(
    phi_g[1:Ny + 1, 1:Nx + 1],
    origin="lower",
    extent=[0.0, Lx, 0.0, Ly],
    cmap="turbo",
    aspect="equal",
)

fig.colorbar(im, ax=ax, label="phi [-]")

title = ax.set_title("")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

im.set_clim(
    min(phi_in_bottom, phi_initial),
    max(phi_in_top, phi_initial),
)


# Time integration
for step in range(nsteps):
    time = (step + 1) * dt

    print(f"Time step {step + 1}/{nsteps}, t = {time:.2f} s", flush=True)

    phi_old = phi_g[1:Ny + 1, 1:Nx + 1].copy()

    phi_new = solve_scalar_field(
        K_inv,
        Su,
        aP0,
        phi_old,
    )

    err_inf = np.max(np.abs(phi_new - phi_old))

    print(f"  err_inf = {err_inf:.6e}", flush=True)

    phi_g[1:Ny + 1, 1:Nx + 1] = phi_new

    fill_ghost_phi(
        phi_g,
        x_c,
        y_c,
        Lx,
        Ly,
        phi_in_top,
        phi_in_bottom,
        y_split,
        top_adiabatic=True,
        bottom_adiabatic=True,
        outlet_zero_grad=True,
    )

    im.set_data(phi_g[1:Ny + 1, 1:Nx + 1])
    title.set_text(f"phi ({scheme.upper()}) at t = {time:.0f} s")
    plt.pause(0.001)

    if err_inf < tolerance:
        print(
            f"Converged: max(|Δphi|) < {tolerance:.1e} "
            f"at t = {time:.2f} s"
        )
        break


# Export final scalar field
write_paraview_structured_grid_vtk(
    Xc,
    Yc,
    phi_g[1:Ny + 1, 1:Nx + 1],
    filename=os.path.join(vtk_dir, f"phi_ghost_{scheme}.vtk"),
    field_name="phi",
)

print(
    "Scalar field exported to:",
    os.path.join(vtk_dir, f"phi_ghost_{scheme}.vtk"),
)


plt.ioff()
plt.show()
