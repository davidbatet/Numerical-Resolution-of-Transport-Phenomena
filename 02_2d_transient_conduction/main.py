"""
2D transient heat conduction in a heterogeneous solid domain.

Numerical method:
    - Cell-centered finite volume method
    - Fully implicit transient formulation
    - Structured Cartesian mesh with optional two-sided tanh refinement
    - Harmonic interpolation of thermal conductivity at internal faces
    - Dense global matrix assembly
    - Precomputed inverse matrix reused during the transient loop

Physical problem:
    rho cp dT/dt = div(k grad(T))

Boundary conditions:
    - Bottom wall: prescribed temperature
    - Top wall: prescribed heat flux
    - Left wall: convective Robin boundary condition
    - Right wall: time-dependent prescribed temperature

Author: David Batet Romero
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from functions import (
    tanh_edges_2sided,
    build_mesh,
    assign_materials,
    assign_coefficients,
    build_global_matrix,
    invert_matrix,
    solve_temperature_field,
    save_mesh_centers,
    write_mesh_structured_grid_vtk,
    write_temperature_structured_grid_vtk,
)


# Problem geometry
Lx = 1.10
Ly = 0.80

x_interface = 0.50
y_interface_1 = 0.40
y_interface_2 = 0.70


# Material properties
rho = {
    "M1": 1500.0,
    "M2": 1600.0,
    "M3": 1900.0,
    "M4": 2500.0,
}

cp = {
    "M1": 750.0,
    "M2": 770.0,
    "M3": 810.0,
    "M4": 930.0,
}

k = {
    "M1": 170.0,
    "M2": 140.0,
    "M3": 200.0,
    "M4": 140.0,
}


# Boundary conditions
T_bottom = 23.0
q_top = 60.0
h_left = 9.0
T_inf_left = 33.0


def T_right(time):
    """
    Time-dependent prescribed temperature at the right boundary.
    """
    return 8.0 + 0.005 * time


# Initial and time parameters
T_initial = 8.0
t_final = 5000.0
snapshot_time = 5000.0
dt = 100.0

nsteps = int(t_final / dt)


# Mesh parameters
Nx = 110
Ny = 80

beta_x = 2.0
beta_y = 3.0


# Output directories
base_dir = os.path.dirname(os.path.abspath(__file__))

results_dir = os.path.join(base_dir, "results")
vtk_dir = os.path.join(base_dir, "vtk")
figures_dir = os.path.join(base_dir, "figures")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(vtk_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)


# Mesh generation
x_edges = tanh_edges_2sided(Lx, Nx, beta_x)
y_edges = tanh_edges_2sided(Ly, Ny, beta_y)

x_edges[0] = 0.0
x_edges[-1] = Lx
y_edges[0] = 0.0
y_edges[-1] = Ly

x_e, y_e, x_c, y_c, dx_cell, dy_cell, Xc, Yc = build_mesh(
    Lx,
    Ly,
    Nx,
    Ny,
    x_edges=x_edges,
    y_edges=y_edges,
)


# Save mesh data
save_mesh_centers(
    x_c,
    y_c,
    filename=os.path.join(results_dir, "mesh_centers.dat"),
)

write_mesh_structured_grid_vtk(
    x_e,
    y_e,
    filename=os.path.join(vtk_dir, "mesh.vtk"),
)

print("Mesh exported to:", os.path.join(vtk_dir, "mesh.vtk"))


# Material assignment
rho_cell, cp_cell, k_cell = assign_materials(
    Xc,
    Yc,
    x_interface,
    y_interface_1,
    y_interface_2,
    rho,
    cp,
    k,
)


# Finite volume coefficients
aE, aW, aN, aS, aP, Su, a_bE_vec = assign_coefficients(
    Nx,
    Ny,
    x_c,
    y_c,
    x_e,
    y_e,
    dx_cell,
    dy_cell,
    k_cell,
    T_bottom,
    q_top,
    h_left,
    T_inf_left,
)


# Transient contribution
cell_area = dy_cell[:, None] * dx_cell[None, :]
aP0 = rho_cell * cp_cell * cell_area / dt
aP_effective = aP + aP0


# Global matrix and inverse
K = build_global_matrix(
    aE,
    aW,
    aN,
    aS,
    aP_effective,
    Nx,
    Ny,
)

K_inv = invert_matrix(K)


# Time integration
T = np.full((Ny, Nx), T_initial, dtype=float)
T_old = T.copy()

T_snapshot = None

for step in range(nsteps):
    time = (step + 1) * dt

    print(f"Time step {step + 1}/{nsteps}, t = {time:.2f} s", flush=True)

    Su_effective = Su.copy()

    # Time-dependent right boundary contribution
    TR = T_right(time)
    Su_effective[:, Nx - 1] += a_bE_vec * TR

    T = solve_temperature_field(
        K_inv,
        Su_effective,
        aP0,
        T_old,
    )

    T_old[:] = T

    if time == snapshot_time:
        T_snapshot = T.copy()

if T_snapshot is None:
    T_snapshot = T.copy()


# Export final temperature field
write_temperature_structured_grid_vtk(
    Xc,
    Yc,
    T,
    filename=os.path.join(vtk_dir, "temperature_field.vtk"),
)

print(
    "Temperature field exported to:",
    os.path.join(vtk_dir, "temperature_field.vtk"),
)


# Plot selected temperature field
fig, ax = plt.subplots(figsize=(7, 4))

pc = ax.pcolormesh(
    x_e,
    y_e,
    T_snapshot,
    cmap="jet",
    shading="auto",
)

fig.colorbar(pc, ax=ax, label="Temperature [°C]")

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"Temperature at t = {snapshot_time:.0f} s")

ax.set_xlim(0.0, Lx)
ax.set_ylim(0.0, Ly)
ax.set_aspect("equal", adjustable="box")

fig.tight_layout()
plt.show()
