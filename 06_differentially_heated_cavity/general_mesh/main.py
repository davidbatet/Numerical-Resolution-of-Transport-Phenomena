"""
2D differentially heated cavity on a general non-uniform MAC grid.

Numerical method:
    - Incompressible Navier-Stokes equations with Boussinesq buoyancy
    - Energy equation for active scalar transport
    - Staggered MAC grid
    - Optional two-sided tanh mesh refinement
    - Fractional-step / projection method
    - AB2 time integration for momentum and temperature
    - First-order upwind treatment for convective terms
    - Non-uniform finite-difference operators
    - Dense pressure Poisson matrix
    - Precomputed inverse matrix reused during the time loop

Benchmark:
    - de Vahl Davis-type post-processing metrics
    - Average and local Nusselt number at the hot and cold walls

Author: David Batet Romero
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from functions import (
    build_mac_grid_tanh,
    create_fields,
    apply_bc_velocity_dhc,
    assemble_K_poisson_neumann_nonuniform,
    invert_K,
    run_solver_dhc_nonuniform,
    compute_cell_center_velocity,
    compute_nusselt_hot_cold_nonuniform,
    compute_de_vahl_davis_metrics,
    print_de_vahl_davis_summary,
    write_vtk_rectilinear_cell_data,
)


# Domain and mesh
xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0

Nx = 60
Ny = 60


# Physical parameters
Ra = 1.0e6
Pr = 0.71

Thot = 1.0
Tcold = 0.0

# Buoyancy reference: "cold" or "mean"
Tref_mode = "cold"


# Time settings
nsteps = 80000

Cconv = 0.35
Cdiff = 0.20
relax = 0.70
dt_cap = 1.0e-4


# Mesh refinement
beta_x = 2.0
beta_y = 2.0


# Convergence criteria
tol_du = 1.0e-6
tol_div = 1.0e-10
tol_dp = 1.0e-6
tol_dT = 1.0e-6
verbose_every = 500


# Plot selector
PLOT = "T"       # "T", "speed", "pressure", "stream", "T+stream", "vorticity", "Nu_hot"
CMAP = "turbo"
LEVELS = 60


# Export settings
EXPORT_VTK = True


# Output directories
base_dir = os.path.dirname(os.path.abspath(__file__))
vtk_dir = os.path.join(base_dir, "vtk")
figures_dir = os.path.join(base_dir, "figures")

os.makedirs(vtk_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)


# Grid and fields
grid = build_mac_grid_tanh(
    xmin,
    xmax,
    ymin,
    ymax,
    Nx,
    Ny,
    beta_x=beta_x,
    beta_y=beta_y,
)

fields = create_fields(Nx, Ny)

x_e = grid["x_e"]
y_e = grid["y_e"]

x_c = grid["x_c"]
y_c = grid["y_c"]

Xc = grid["Xc"]
Yc = grid["Yc"]

dx_cell = grid["dx_cell"]
dy_cell = grid["dy_cell"]


# Initial condition
fields["T"][:, :] = Thot + (Tcold - Thot) * Xc
fields["u"][:, :] = 0.0
fields["v"][:, :] = 0.0
fields["p"][:, :] = 0.0

apply_bc_velocity_dhc(fields["u"], fields["v"])


# Constant time-step estimation
dx_min = float(np.min(dx_cell))
dy_min = float(np.min(dy_cell))

umax0 = 1.0e-8
vmax0 = 1.0e-8

dt_conv = Cconv * min(
    dx_min / max(umax0, 1.0e-14),
    dy_min / max(vmax0, 1.0e-14),
)

dt_diff_u = Cdiff * min(dx_min * dx_min, dy_min * dy_min) / max(Pr, 1.0e-14)
dt_diff_T = Cdiff * min(dx_min * dx_min, dy_min * dy_min)

dt = relax * min(dt_cap, dt_conv, dt_diff_u, dt_diff_T)

print(f"dt_conv   = {dt_conv:.6e}")
print(f"dt_diff_u = {dt_diff_u:.6e}")
print(f"dt_diff_T = {dt_diff_T:.6e}")
print(f"dt_used   = {dt:.6e}")


# Pressure Poisson matrix
K = assemble_K_poisson_neumann_nonuniform(
    x_e=x_e,
    x_c=x_c,
    y_e=y_e,
    y_c=y_c,
    pin_reference=True,
    ref_i=0,
    ref_j=0,
)

print("Inverting pressure Poisson matrix...", flush=True)
K_inv = invert_K(K)
print("Starting differentially heated cavity solver...", flush=True)


# Solver
fields, history = run_solver_dhc_nonuniform(
    fields=fields,
    grid=grid,
    Ra=Ra,
    Pr=Pr,
    Thot=Thot,
    Tcold=Tcold,
    Tref_mode=Tref_mode,
    dt=dt,
    nsteps=nsteps,
    K_inv=K_inv,
    tol_du=tol_du,
    tol_div=tol_div,
    tol_dp=tol_dp,
    tol_dT=tol_dT,
    verbose_every=verbose_every,
)


# Cell-centered fields
p = fields["p"]
T = fields["T"]

u_center, v_center = compute_cell_center_velocity(fields, Nx, Ny)
speed = np.sqrt(u_center**2 + v_center**2)


# Nusselt number and benchmark-style metrics
Nu_hot_local, Nu_hot_avg, Nu_cold_local, Nu_cold_avg = (
    compute_nusselt_hot_cold_nonuniform(
        T=T,
        x_e=x_e,
        x_c=x_c,
        Thot=Thot,
        Tcold=Tcold,
    )
)

metrics = compute_de_vahl_davis_metrics(
    fields,
    grid,
    Thot=Thot,
    Tcold=Tcold,
)

print_de_vahl_davis_summary(
    metrics,
    Ra=Ra,
    Pr=Pr,
    Nx=Nx,
    Ny=Ny,
)

print("\n--- Nusselt number summary ---")
print(f"Nu_hot_avg  = {Nu_hot_avg:.6f}")
print(f"Nu_cold_avg = {Nu_cold_avg:.6f}")


# VTK export
if EXPORT_VTK:
    velocity = np.zeros((Ny, Nx, 2), dtype=float)
    velocity[:, :, 0] = u_center
    velocity[:, :, 1] = v_center

    Nu_hot_map = np.tile(Nu_hot_local.reshape(Ny, 1), (1, Nx))

    vtk_path = os.path.join(vtk_dir, "differentially_heated_cavity_general.vtk")

    write_vtk_rectilinear_cell_data(
        filename=vtk_path,
        x_e=x_e,
        y_e=y_e,
        cell_data={
            "pressure": p,
            "temperature": T,
            "u_center": u_center,
            "v_center": v_center,
            "speed": speed,
            "velocity": velocity,
            "Nu_hot_profile": Nu_hot_map,
        },
    )

    print(f"VTK field exported to: {vtk_path}")


# Plot selected field
plt.figure(figsize=(6.6, 5.4))

if PLOT == "T":
    plt.contourf(Xc, Yc, T, LEVELS, cmap=CMAP)
    plt.colorbar(label="T")

elif PLOT == "speed":
    plt.contourf(Xc, Yc, speed, LEVELS, cmap=CMAP)
    plt.colorbar(label="|u|")

elif PLOT == "pressure":
    plt.contourf(Xc, Yc, p, LEVELS, cmap=CMAP)
    plt.colorbar(label="p")

elif PLOT == "stream":
    plt.streamplot(x_c, y_c, u_center, v_center, density=1.6)

elif PLOT == "T+stream":
    plt.contourf(Xc, Yc, T, LEVELS, cmap=CMAP)
    plt.colorbar(label="T")
    plt.streamplot(x_c, y_c, u_center, v_center, density=1.3, linewidth=0.9)

elif PLOT == "vorticity":
    dudy, dudx = np.gradient(u_center, y_c, x_c, edge_order=2)
    dvdy, dvdx = np.gradient(v_center, y_c, x_c, edge_order=2)

    omega = dvdx - dudy

    plt.contourf(Xc, Yc, omega, LEVELS, cmap="RdBu_r")
    plt.colorbar(label="omega")

elif PLOT == "Nu_hot":
    plt.plot(Nu_hot_local, y_c, "-k")
    plt.xlabel("Nu_hot(y)")
    plt.ylabel("y")
    plt.grid(True)
    plt.title(f"Local hot-wall Nusselt number, average = {Nu_hot_avg:.6f}")

else:
    raise ValueError(f"Unknown PLOT option: {PLOT}")

if PLOT != "Nu_hot":
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        f"DHC general mesh, tanh beta=({beta_x}, {beta_y}), "
        f"Ra={Ra:g}, Pr={Pr:g}, plot={PLOT}"
    )

plt.tight_layout()
plt.show()
