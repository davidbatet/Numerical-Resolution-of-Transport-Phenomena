"""
Smith-Hutton convection-diffusion benchmark problem on a uniform mesh.

Numerical method:
    - Cell-centered finite volume method
    - Patankar-style boundary coefficient treatment
    - Upwind or power-law convection-diffusion scheme
    - Pseudo-transient iterative formulation
    - Uniform structured Cartesian mesh
    - Dense global matrix assembly
    - Precomputed inverse matrix reused during the pseudo-time loop

Physical problem:
    div(rho u phi) = div(Gamma grad(phi))

Velocity field:
    u =  2y(1 - x^2)
    v = -2x(1 - y^2)

Author: David Batet Romero
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from functions import (
    build_mesh,
    assemble_matrix_and_source_smith_hutton,
    invert_matrix,
    solve_interior_field,
    write_paraview_structured_grid_vtk,
)


# Smith-Hutton domain
xmin = -1.0
xmax = 1.0
ymin = 0.0
ymax = 1.0

Nx = 100
Ny = 100


# Physical parameters
alpha = 10.0
rho_over_Gamma = 1.0e3

Gamma = 1.0e-9
rho = rho_over_Gamma * Gamma


# Numerical scheme
scheme = "powerlaw"  # Available options: "upwind", "powerlaw"


# Pseudo-time parameters
dt = 0.1
t_final = 1000.0
nsteps = int(t_final / dt)

tolerance = 1.0e-6


# Output directories
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")
vtk_dir = os.path.join(base_dir, "vtk")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(vtk_dir, exist_ok=True)


# Mesh generation
mesh = build_mesh(
    xmin,
    xmax,
    ymin,
    ymax,
    Nx,
    Ny,
)

x_e = mesh["x_e"]
y_e = mesh["y_e"]

x_c = mesh["x_c"]
y_c = mesh["y_c"]

dx_cell = mesh["dx_cell"]
dy_cell = mesh["dy_cell"]

Xc = mesh["Xc"]
Yc = mesh["Yc"]


# Pseudo-transient coefficient
cell_area = dy_cell[:, None] * dx_cell[None, :]
aP0 = rho * cell_area / dt


# Initial condition
phi_else = 1.0 - np.tanh(alpha)
phi = np.full((Ny, Nx), phi_else, dtype=float)


# Matrix and source assembly
K, Su = assemble_matrix_and_source_smith_hutton(
    Nx,
    Ny,
    x_e,
    y_e,
    x_c,
    y_c,
    dx_cell,
    dy_cell,
    Gamma=Gamma,
    rho=rho,
    alpha=alpha,
    aP0=aP0,
    scheme=scheme,
)


# Matrix inversion
print("Inverting global matrix...", flush=True)
K_inv = invert_matrix(K)
print("Starting pseudo-time integration...", flush=True)


# Live plot
plt.ion()

fig, ax = plt.subplots(figsize=(8.0, 4.2))

im = ax.imshow(
    phi,
    origin="lower",
    extent=[xmin, xmax, ymin, ymax],
    cmap="jet",
    aspect="auto",
)

fig.colorbar(im, ax=ax, label="phi [-]")

ax.set_xlabel("x")
ax.set_ylabel("y")

title = ax.set_title("")

im.set_clim(phi_else, 2.0)


# Pseudo-time loop
for step in range(nsteps):
    time = (step + 1) * dt

    phi_old = phi.copy()

    phi_new = solve_interior_field(
        K_inv,
        Su,
        aP0,
        phi_old,
    )

    err_inf = np.max(np.abs(phi_new - phi_old))

    phi = phi_new

    if step % 50 == 0 or err_inf < tolerance:
        print(
            f"step = {step:6d}  "
            f"t = {time:10.4f}  "
            f"err_inf = {err_inf:.3e}",
            flush=True,
        )

        im.set_data(phi)

        title.set_text(
            f"Smith-Hutton  rho/Gamma={rho_over_Gamma:.0e}  "
            f"{scheme.upper()}  t={time:.2f}"
        )

        plt.pause(0.001)

    if err_inf < tolerance:
        print(f"Converged: max(|dphi|) < {tolerance:.1e} at t = {time:.4f}")
        break


plt.ioff()
plt.show()


# Outlet sampling
x_ref = np.linspace(0.0, 1.0, 11)

j0 = 0
j1 = 1

y0 = y_c[j0]
y1 = y_c[j1]

phi_out = np.zeros_like(x_ref)

for k, xr in enumerate(x_ref):
    i = int(np.argmin(np.abs(x_c - xr)))

    phi0 = phi[j0, i]
    phi1 = phi[j1, i]

    # Linear interpolation to y = 0
    phi_out[k] = phi0 + (0.0 - y0) * (phi1 - phi0) / (y1 - y0)


print("\nOutlet values sampled at y = 0 using linear interpolation")
print(f"y0 = {y0:.6e}, y1 = {y1:.6e}")
print("x_ref    phi_out")

for xr, value in zip(x_ref, phi_out):
    print(f"{xr:4.1f}    {value: .6f}")


# Save outlet samples
csv_name = f"outlet_phi_{scheme}_rhoGamma_{rho_over_Gamma:.0e}.csv"
csv_path = os.path.join(results_dir, csv_name)

data = np.column_stack([x_ref, phi_out])

np.savetxt(
    csv_path,
    data,
    delimiter=",",
    header="x,phi",
    comments="",
)

print(f"\nOutlet samples saved to: {csv_path}")


# VTK export
vtk_name = f"phi_smith_hutton_{scheme}_rhoGamma_{rho_over_Gamma:.0e}.vtk"
vtk_path = os.path.join(vtk_dir, vtk_name)

write_paraview_structured_grid_vtk(
    Xc,
    Yc,
    phi,
    vtk_path,
    field_name="phi",
)

print(f"VTK field exported to: {vtk_path}")
