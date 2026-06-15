"""
2D lid-driven cavity flow on a uniform MAC grid.

Numerical method:
    - Incompressible Navier-Stokes equations
    - Staggered MAC grid
    - Fractional-step / projection method
    - AB2 time integration for the momentum predictor
    - First-order upwind treatment for convective fluxes
    - Central differences for diffusion
    - Dense pressure Poisson matrix
    - Precomputed inverse matrix reused during the time loop

Benchmark:
    - Optional comparison with Ghia et al. centerline data for Re = 100

Author: David Batet Romero
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from functions import (
    build_mac_grid,
    create_fields,
    assemble_K_poisson_neumann,
    invert_K,
    run_solver_lid_driven_cavity,
    compute_cell_center_velocity,
    write_vtk_rectilinear_cell_data,
)


# Domain and mesh
xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0

Nx = 70
Ny = 70


# Physical parameters
Re = 1000.0
U_lid = 1.0
nu = 1.0 / Re


# Time settings
nsteps = 25000

dt_user = 1.0e-1
Cconv = 0.35
Cvisc = 0.20
relax = 0.50


# Convergence criteria
tol_du = 1.0e-8
tol_div = 1.0e-8
tol_dp = 1.0e-8
verbose_every = 100


# Plot selector
PLOT = "v"       # "speed", "pressure", "u", "v", "stream", "speed+stream", "quiver", "vorticity"
CMAP = "turbo"  # "turbo", "jet", "viridis", "plasma", "inferno", "magma", "cividis"
LEVELS = 60


# Export settings
EXPORT_VTK = True
RUN_GHIA_RE100_VALIDATION = True


# Output directories
base_dir = os.path.dirname(os.path.abspath(__file__))
vtk_dir = os.path.join(base_dir, "vtk")
figures_dir = os.path.join(base_dir, "figures")

os.makedirs(vtk_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)


# Grid and fields
grid = build_mac_grid(xmin, xmax, ymin, ymax, Nx, Ny)
fields = create_fields(Nx, Ny)

x_e = grid["x_e"]
y_e = grid["y_e"]
x_c = grid["x_c"]
y_c = grid["y_c"]
Xc = grid["Xc"]
Yc = grid["Yc"]

dx = grid["dx"]
dy = grid["dy"]


# Time-step estimation
umax = U_lid
vmax = U_lid

dt_conv = Cconv * min(
    dx / max(umax, 1.0e-14),
    dy / max(vmax, 1.0e-14),
)

dt_visc = Cvisc * min(dx * dx, dy * dy) / nu

dt = relax * min(dt_user, dt_conv, dt_visc)

print(f"dt_conv = {dt_conv:.6e}")
print(f"dt_visc = {dt_visc:.6e}")
print(f"dt_used = {dt:.6e}")


# Pressure Poisson matrix
K = assemble_K_poisson_neumann(
    Nx,
    Ny,
    dx,
    dy,
    pin_reference=True,
    ref_i=0,
    ref_j=0,
)

print("Inverting pressure Poisson matrix...", flush=True)
K_inv = invert_K(K)
print("Starting lid-driven cavity solver...", flush=True)


# Solver
fields = run_solver_lid_driven_cavity(
    fields=fields,
    grid=grid,
    Re=Re,
    dt=dt,
    nsteps=nsteps,
    K_inv=K_inv,
    U_lid=U_lid,
    tol_du=tol_du,
    tol_div=tol_div,
    tol_dp=tol_dp,
    verbose_every=verbose_every,
)


# Cell-centered fields
p = fields["p"]

u_center, v_center = compute_cell_center_velocity(fields, Nx, Ny)
speed = np.sqrt(u_center**2 + v_center**2)


# VTK export
if EXPORT_VTK:
    velocity = np.zeros((Ny, Nx, 2), dtype=float)
    velocity[:, :, 0] = u_center
    velocity[:, :, 1] = v_center

    vtk_path = os.path.join(vtk_dir, "lid_driven_cavity_uniform.vtk")

    write_vtk_rectilinear_cell_data(
        filename=vtk_path,
        x_e=x_e,
        y_e=y_e,
        cell_data={
            "pressure": p,
            "u_center": u_center,
            "v_center": v_center,
            "speed": speed,
            "velocity": velocity,
        },
    )

    print(f"VTK field exported to: {vtk_path}")


# Plot selected field
plt.figure(figsize=(6.3, 5.3))

if PLOT == "speed":
    plt.contourf(Xc, Yc, speed, LEVELS, cmap=CMAP)
    plt.colorbar(label="|u|")

elif PLOT == "pressure":
    plt.contourf(Xc, Yc, p, LEVELS, cmap=CMAP)
    plt.colorbar(label="p")

elif PLOT == "u":
    plt.contourf(Xc, Yc, u_center, LEVELS, cmap="coolwarm")
    plt.colorbar(label="u")

elif PLOT == "v":
    plt.contourf(Xc, Yc, v_center, LEVELS, cmap="coolwarm")
    plt.colorbar(label="v")

elif PLOT == "stream":
    plt.streamplot(x_c, y_c, u_center, v_center, density=1.6)

elif PLOT == "speed+stream":
    plt.contourf(Xc, Yc, speed, LEVELS, cmap=CMAP)
    plt.colorbar(label="|u|")
    plt.streamplot(x_c, y_c, u_center, v_center, density=1.3)

elif PLOT == "quiver":
    plt.contourf(Xc, Yc, speed, LEVELS, cmap=CMAP)
    plt.colorbar(label="|u|")

    step = max(1, Nx // 25)

    plt.quiver(
        Xc[::step, ::step],
        Yc[::step, ::step],
        u_center[::step, ::step],
        v_center[::step, ::step],
    )

elif PLOT == "vorticity":
    dvdx = (v_center[:, 1:] - v_center[:, :-1]) / dx
    dudy = (u_center[1:, :] - u_center[:-1, :]) / dy

    omega = dvdx[:-1, :] - dudy[:, :-1]

    plt.contourf(Xc[:-1, :-1], Yc[:-1, :-1], omega, LEVELS, cmap="RdBu_r")
    plt.colorbar(label="omega")

else:
    raise ValueError(f"Unknown PLOT option: {PLOT}")

plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Lid-driven cavity uniform mesh, Re = {Re:g}, plot = {PLOT}")
plt.tight_layout()
plt.show()


def ghia_re100():
    """
    Ghia et al. benchmark data for Re = 100.

    Returns:
        xg, vg: v velocity along y = 0.5
        yg, ug: u velocity along x = 0.5
    """
    yg = np.array([
        1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516,
        0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719,
        0.1016, 0.0703, 0.0625, 0.0547, 0.0000,
    ])

    ug = np.array([
        1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151,
        0.00332, -0.13641, -0.20581, -0.21090, -0.15662,
        -0.10150, -0.06434, -0.04775, -0.04192, -0.03717,
        0.00000,
    ])

    xg = np.array([
        1.00000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063,
        0.8594, 0.8047, 0.5000, 0.2344, 0.2266, 0.1563,
        0.0938, 0.0781, 0.0703, 0.0625, 0.0000,
    ])

    vg = np.array([
        0.00000, -0.05906, -0.07391, -0.08864, -0.10313,
        -0.16914, -0.22445, -0.24533, 0.05454, 0.17527,
        0.17507, 0.16077, 0.12317, 0.10890, 0.10091,
        0.09233, 0.00000,
    ])

    return xg, vg, yg, ug


# Optional validation against Ghia Re = 100 data
if RUN_GHIA_RE100_VALIDATION:
    if not np.isclose(Re, 100.0):
        print(
            "\nGhia Re=100 validation skipped because the current simulation "
            f"uses Re = {Re:g}."
        )
        print(
            "Set Re = 100.0 to compare against the included Ghia Re=100 data."
        )

    else:
        x_target = 0.5
        y_target = 0.5

        u_x05 = np.array([
            np.interp(x_target, x_c, u_center[j, :])
            for j in range(len(y_c))
        ])

        y_full_u = np.r_[0.0, y_c, 1.0]
        u_full = np.r_[0.0, u_x05, U_lid]

        v_y05 = np.array([
            np.interp(y_target, y_c, v_center[:, i])
            for i in range(len(x_c))
        ])

        x_full_v = np.r_[0.0, x_c, 1.0]
        v_full = np.r_[0.0, v_y05, 0.0]

        xg, vg, yg, ug = ghia_re100()

        ord_y = np.argsort(yg)
        yg_sorted = yg[ord_y]
        ug_sorted = ug[ord_y]

        ord_x = np.argsort(xg)
        xg_sorted = xg[ord_x]
        vg_sorted = vg[ord_x]

        u_interp = np.interp(yg_sorted, y_full_u, u_full)
        v_interp = np.interp(xg_sorted, x_full_v, v_full)

        u_error = u_interp - ug_sorted
        v_error = v_interp - vg_sorted

        print("\n--- Ghia Re=100 centerline comparison ---")
        print(f"max |u error| = {np.max(np.abs(u_error)):.3e}")
        print(f"max |v error| = {np.max(np.abs(v_error)):.3e}")

        plt.figure(figsize=(6.0, 4.5))
        plt.plot(u_full, y_full_u, "-", label="Computed")
        plt.plot(ug_sorted, yg_sorted, "o", label="Ghia et al.")
        plt.xlabel("u(x = 0.5, y)")
        plt.ylabel("y")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(6.0, 4.5))
        plt.plot(x_full_v, v_full, "-", label="Computed")
        plt.plot(xg_sorted, vg_sorted, "o", label="Ghia et al.")
        plt.xlabel("x")
        plt.ylabel("v(x, y = 0.5)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.show()
