"""
Smith-Hutton convection-diffusion benchmark problem.

Numerical method:
    - Cell-centered finite volume method
    - Patankar-style boundary coefficient treatment
    - Upwind or power-law convection-diffusion scheme
    - Pseudo-transient iterative formulation
    - Structured Cartesian mesh with optional one-sided tanh refinement in y
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
    tanh_edges_onesided,
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

Nx = 50
Ny = 50


# Physical parameters
alpha = 10.0
rho_over_Gamma = 1.0e1

Gamma = 1.0
rho = rho_over_Gamma * Gamma


# Numerical scheme
scheme = "powerlaw"  # Available options: "upwind", "powerlaw"


# Pseudo-time parameters
dt = 1.0
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
beta_y = 0.0

y_edges = tanh_edges_onesided(ymin, ymax, Ny, beta_y)
y_edges[0] = ymin
y_edges[-1] = ymax

mesh = build_mesh(
    xmin,
    xmax,
    ymin,
    ymax,
    Nx,
    Ny,
    y_edges=y_edges,
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

pc = ax.pcolormesh(
    x_e,
    y_e,
    phi,
    shading="auto",
    cmap="jet",
)

fig.colorbar(pc, ax=ax, label="phi [-]")

ax.set_xlabel("x")
ax.set_ylabel("y")

title = ax.set_title("")

pc.set_clim(phi_else, 2.0)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)


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

        pc.set_array(phi.ravel(order="C"))

        title.set_text(
            f"Smith-Hutton  rho/Gamma={rho_over_Gamma:.0e}  "
            f"{scheme.upper()}  t={time:.2f}"
        )

        fig.canvas.draw_idle()
        plt.pause(0.001)

    if err_inf < tolerance:
        print(f"Converged: max(|dphi|) < {tolerance:.1e} at t = {time:.4f}")
        break


plt.ioff()
plt.show()


# Outlet sampling
x_ref = np.linspace(0.0, 1.0, 11)

j_out = 0
y_out = y_c[j_out]

phi_line = phi[j_out, :]
phi_out = np.interp(x_ref, x_c, phi_line)

print("\nOutlet values sampled near y = 0 using the first cell-center row")
print(f"y_out = {y_out:.6e}")
print("x_ref    phi_out")

for xr, value in zip(x_ref, phi_out):
    print(f"{xr:4.1f}    {value: .6f}")


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


# Benchmark comparison
BENCHMARK_EXPECTED = {
    1.0e1: np.array(
        [1.989, 1.402, 1.146, 0.946, 0.775, 0.621,
         0.480, 0.349, 0.227, 0.111, 0.000]
    ),
    1.0e3: np.array(
        [2.0000, 1.9990, 1.9997, 1.9850, 1.8410, 0.9510,
         0.1540, 0.0010, 0.0000, 0.0000, 0.0000]
    ),
    1.0e6: np.array(
        [2.000, 2.000, 2.000, 1.999, 1.964, 1.000,
         0.036, 0.001, 0.000, 0.000, 0.000]
    ),
}


def compare_with_benchmark(rho_over_Gamma_value, x_ref, phi_out):
    """
    Compare the computed outlet profile with tabulated Smith-Hutton values.
    """
    keys = np.array(list(BENCHMARK_EXPECTED.keys()), dtype=float)
    benchmark_key = float(keys[np.argmin(np.abs(keys - rho_over_Gamma_value))])

    expected = BENCHMARK_EXPECTED[benchmark_key]

    if len(expected) != len(phi_out):
        raise ValueError("Benchmark and computed outlet arrays have different lengths.")

    abs_error = np.abs(phi_out - expected)

    rel_error = np.zeros_like(abs_error)
    mask = np.abs(expected) > 1.0e-14
    rel_error[mask] = abs_error[mask] / np.abs(expected[mask])

    print("\n--- Smith-Hutton outlet comparison ---")
    print(
        f"rho/Gamma = {rho_over_Gamma_value:.0e} "
        f"(benchmark key used: {benchmark_key:.0e})"
    )
    print("   x      expected     calculated     abs_err     rel_err")

    for xr, expv, calv, ae, re in zip(
        x_ref,
        expected,
        phi_out,
        abs_error,
        rel_error,
    ):
        print(f" {xr:4.1f}   {expv: .6f}    {calv: .6f}   {ae: .3e}   {re: .3e}")

    print(f"\nmax abs error  = {abs_error.max():.3e}")
    print(f"mean abs error = {abs_error.mean():.3e}")

    return expected, abs_error, rel_error


expected_out, abs_error, rel_error = compare_with_benchmark(
    rho_over_Gamma,
    x_ref,
    phi_out,
)


# Outlet profile plot
plt.figure(figsize=(7.5, 4.2))

plt.plot(
    x_ref,
    expected_out,
    marker="o",
    linewidth=1.5,
    label="Benchmark",
)

plt.plot(
    x_ref,
    phi_out,
    marker="s",
    linewidth=1.5,
    label="Computed",
)

plt.xlabel("x at outlet near y = 0")
plt.ylabel("phi [-]")
plt.title(
    f"Smith-Hutton outlet profile: "
    f"rho/Gamma={rho_over_Gamma:.0e}, scheme={scheme.upper()}"
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# Absolute error plot
plt.figure(figsize=(7.5, 4.2))

plt.plot(
    x_ref,
    abs_error,
    marker="o",
    linewidth=1.5,
)

plt.xlabel("x at outlet near y = 0")
plt.ylabel("|phi_computed - phi_benchmark|")
plt.title(
    f"Outlet absolute error: "
    f"rho/Gamma={rho_over_Gamma:.0e}, scheme={scheme.upper()}"
)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
