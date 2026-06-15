"""
Finite volume functions for the 2D transient heat conduction problem.

The implementation follows the numerical procedure developed in the thesis:
    - Cell-centered finite volume discretization
    - Fully implicit transient term
    - Dense matrix assembly
    - Precomputed matrix inverse reused at each time step

Author: David Batet Romero
"""

import numpy as np


def tanh_edges_2sided(L, N, beta):
    """
    Generate a two-sided hyperbolic tangent mesh distribution.

    For beta <= 0, a uniform mesh is returned.
    """
    if beta is None or beta <= 0.0:
        return np.linspace(0.0, L, N + 1)

    xi = np.linspace(0.0, 1.0, N + 1)
    s = 2.0 * xi - 1.0

    x_edges = 0.5 * L * (1.0 + np.tanh(beta * s) / np.tanh(beta))

    return x_edges


def build_mesh(Lx, Ly, Nx, Ny, x_edges=None, y_edges=None):
    """
    Build a structured Cartesian cell-centered mesh.
    """
    if x_edges is None:
        x_e = np.linspace(0.0, Lx, Nx + 1)
    else:
        x_e = np.asarray(x_edges, dtype=float)
        if len(x_e) != Nx + 1:
            raise ValueError("x_edges must contain Nx + 1 points.")

    if y_edges is None:
        y_e = np.linspace(0.0, Ly, Ny + 1)
    else:
        y_e = np.asarray(y_edges, dtype=float)
        if len(y_e) != Ny + 1:
            raise ValueError("y_edges must contain Ny + 1 points.")

    dx_cell = x_e[1:] - x_e[:-1]
    dy_cell = y_e[1:] - y_e[:-1]

    x_c = 0.5 * (x_e[1:] + x_e[:-1])
    y_c = 0.5 * (y_e[1:] + y_e[:-1])

    Xc, Yc = np.meshgrid(x_c, y_c, indexing="xy")

    return x_e, y_e, x_c, y_c, dx_cell, dy_cell, Xc, Yc


def assign_materials(
    Xc,
    Yc,
    x_interface,
    y_interface_1,
    y_interface_2,
    rho,
    cp,
    k,
):
    """
    Assign material properties to each control volume.

    The computational domain is divided into four material regions.
    """
    Ny, Nx = Xc.shape

    rho_cell = np.zeros((Ny, Nx), dtype=float)
    cp_cell = np.zeros((Ny, Nx), dtype=float)
    k_cell = np.zeros((Ny, Nx), dtype=float)

    mask_M1 = (Xc <= x_interface) & (Yc <= y_interface_1)
    mask_M2 = (Xc > x_interface) & (Yc <= y_interface_2)
    mask_M3 = (Xc <= x_interface) & (Yc > y_interface_1)
    mask_M4 = (Xc > x_interface) & (Yc > y_interface_2)

    rho_cell[mask_M1] = rho["M1"]
    rho_cell[mask_M2] = rho["M2"]
    rho_cell[mask_M3] = rho["M3"]
    rho_cell[mask_M4] = rho["M4"]

    cp_cell[mask_M1] = cp["M1"]
    cp_cell[mask_M2] = cp["M2"]
    cp_cell[mask_M3] = cp["M3"]
    cp_cell[mask_M4] = cp["M4"]

    k_cell[mask_M1] = k["M1"]
    k_cell[mask_M2] = k["M2"]
    k_cell[mask_M3] = k["M3"]
    k_cell[mask_M4] = k["M4"]

    if np.any(k_cell <= 0.0):
        raise ValueError("Some control volumes were not assigned a valid material.")

    return rho_cell, cp_cell, k_cell


def assign_coefficients(
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
):
    """
    Assemble finite volume coefficients for the 2D heat conduction problem.

    Boundary conditions:
        - Bottom: Dirichlet temperature
        - Top: prescribed heat flux
        - Left: Robin / convective condition
        - Right: time-dependent Dirichlet temperature

    The right boundary source contribution is added later in the time loop.
    """
    aE = np.zeros((Ny, Nx), dtype=float)
    aW = np.zeros((Ny, Nx), dtype=float)
    aN = np.zeros((Ny, Nx), dtype=float)
    aS = np.zeros((Ny, Nx), dtype=float)
    aP = np.zeros((Ny, Nx), dtype=float)
    Su = np.zeros((Ny, Nx), dtype=float)

    dx_centers = x_c[1:] - x_c[:-1]
    dy_centers = y_c[1:] - y_c[:-1]

    dx_E = np.zeros(Nx, dtype=float)
    dx_W = np.zeros(Nx, dtype=float)

    dx_E[:-1] = dx_centers
    dx_W[1:] = dx_centers

    dy_N = np.zeros(Ny, dtype=float)
    dy_S = np.zeros(Ny, dtype=float)

    dy_N[:-1] = dy_centers
    dy_S[1:] = dy_centers

    # Internal diffusion coefficients
    for j in range(Ny):
        for i in range(Nx):
            kP = k_cell[j, i]

            Ae = dy_cell[j]
            Aw = dy_cell[j]
            An = dx_cell[i]
            As = dx_cell[i]

            if i < Nx - 1:
                kE = k_cell[j, i + 1]
                k_e = 2.0 * kP * kE / (kP + kE)
                aE[j, i] = k_e * Ae / dx_E[i]

            if i > 0:
                kW = k_cell[j, i - 1]
                k_w = 2.0 * kP * kW / (kP + kW)
                aW[j, i] = k_w * Aw / dx_W[i]

            if j < Ny - 1:
                kN = k_cell[j + 1, i]
                k_n = 2.0 * kP * kN / (kP + kN)
                aN[j, i] = k_n * An / dy_N[j]

            if j > 0:
                kS = k_cell[j - 1, i]
                k_s = 2.0 * kP * kS / (kP + kS)
                aS[j, i] = k_s * As / dy_S[j]

    aP = aE + aW + aN + aS

    # Bottom boundary: prescribed temperature
    j = 0
    delta_s = y_c[0] - y_e[0]

    for i in range(Nx):
        kP = k_cell[j, i]
        As = dx_cell[i]

        a_bS = kP * As / delta_s

        aS[j, i] = 0.0
        aP[j, i] += a_bS
        Su[j, i] += a_bS * T_bottom

    # Top boundary: prescribed heat flux
    j = Ny - 1

    for i in range(Nx):
        An = dx_cell[i]

        aN[j, i] = 0.0
        Su[j, i] += q_top * An

    # Left boundary: Robin / convective condition
    i = 0
    delta_w = x_c[0] - x_e[0]

    for j in range(Ny):
        kP = k_cell[j, i]
        Aw = dy_cell[j]

        a_bW = (kP * h_left * Aw) / (kP + h_left * delta_w)

        aW[j, i] = 0.0
        aP[j, i] += a_bW
        Su[j, i] += a_bW * T_inf_left

    # Right boundary: time-dependent prescribed temperature
    i = Nx - 1
    delta_e = x_e[-1] - x_c[-1]

    a_bE_vec = np.zeros(Ny, dtype=float)

    for j in range(Ny):
        kP = k_cell[j, i]
        Ae = dy_cell[j]

        a_bE_vec[j] = kP * Ae / delta_e

        aE[j, i] = 0.0
        aP[j, i] += a_bE_vec[j]

    return aE, aW, aN, aS, aP, Su, a_bE_vec


def build_global_matrix(aE, aW, aN, aS, aP_effective, Nx, Ny):
    """
    Build the dense global coefficient matrix.
    """
    n_cells = Nx * Ny
    K = np.zeros((n_cells, n_cells), dtype=float)

    def index(i, j):
        return j * Nx + i

    for j in range(Ny):
        for i in range(Nx):
            P = index(i, j)

            K[P, P] = aP_effective[j, i]

            if i < Nx - 1:
                E = index(i + 1, j)
                K[P, E] = -aE[j, i]

            if i > 0:
                W = index(i - 1, j)
                K[P, W] = -aW[j, i]

            if j < Ny - 1:
                N = index(i, j + 1)
                K[P, N] = -aN[j, i]

            if j > 0:
                S = index(i, j - 1)
                K[P, S] = -aS[j, i]

    return K


def build_rhs(Su_effective, aP0, T_old):
    """
    Build the right-hand-side vector of the implicit system.
    """
    b = Su_effective + aP0 * T_old

    return b.ravel(order="C")


def invert_matrix(K):
    """
    Compute the inverse of the constant coefficient matrix.

    The inverse is computed once before the transient loop and reused at each
    time step, following the original implementation strategy.
    """
    return np.linalg.inv(K)


def solve_temperature_field(K_inv, Su_effective, aP0, T_old):
    """
    Solve the implicit temperature update using the precomputed inverse matrix.
    """
    Ny, Nx = Su_effective.shape

    b_vec = build_rhs(Su_effective, aP0, T_old)
    T_vec = K_inv @ b_vec

    return T_vec.reshape((Ny, Nx), order="C")


def save_mesh_centers(x_c, y_c, filename="mesh_centers.dat"):
    """
    Save the cell-center coordinates to a text file.
    """
    Xc, Yc = np.meshgrid(x_c, y_c, indexing="xy")

    coordinates = np.column_stack(
        [
            Xc.ravel(order="C"),
            Yc.ravel(order="C"),
        ]
    )

    np.savetxt(filename, coordinates, header="x y")


def write_temperature_structured_grid_vtk(
    Xc,
    Yc,
    T,
    filename,
    field_name="Temperature",
):
    """
    Write the temperature field as a VTK legacy STRUCTURED_GRID file.

    The exported points correspond to cell centers.
    """
    Xc = np.asarray(Xc)
    Yc = np.asarray(Yc)
    T = np.asarray(T)

    if Xc.ndim != 2 or Yc.ndim != 2 or T.ndim != 2:
        raise ValueError("Xc, Yc and T must be 2D arrays.")

    if Xc.shape != Yc.shape or Xc.shape != T.shape:
        raise ValueError(
            f"Shapes must match: Xc={Xc.shape}, Yc={Yc.shape}, T={T.shape}"
        )

    ny, nx = Xc.shape
    npoints = nx * ny

    x_flat = Xc.ravel(order="C").astype(np.float64, copy=False)
    y_flat = Yc.ravel(order="C").astype(np.float64, copy=False)
    T_flat = T.ravel(order="C").astype(np.float64, copy=False)

    with open(filename, "w", encoding="utf-8") as file:
        file.write("# vtk DataFile Version 3.0\n")
        file.write(f"{field_name} on 2D structured grid\n")
        file.write("ASCII\n")
        file.write("DATASET STRUCTURED_GRID\n")
        file.write(f"DIMENSIONS {nx} {ny} 1\n")
        file.write(f"POINTS {npoints} double\n")

        for point in range(npoints):
            file.write(
                f"{x_flat[point]:.16e} "
                f"{y_flat[point]:.16e} "
                f"{0.0:.16e}\n"
            )

        file.write(f"\nPOINT_DATA {npoints}\n")
        file.write(f"SCALARS {field_name} double 1\n")
        file.write("LOOKUP_TABLE default\n")

        for value in T_flat:
            file.write(f"{value:.16e}\n")


def write_mesh_structured_grid_vtk(x_edges, y_edges, filename="mesh.vtk"):
    """
    Write the structured mesh as a VTK legacy STRUCTURED_GRID file.
    """
    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)

    nx = len(x_edges)
    ny = len(y_edges)
    npoints = nx * ny

    with open(filename, "w", encoding="utf-8") as file:
        file.write("# vtk DataFile Version 3.0\n")
        file.write("Structured mesh\n")
        file.write("ASCII\n")
        file.write("DATASET STRUCTURED_GRID\n")
        file.write(f"DIMENSIONS {nx} {ny} 1\n")
        file.write(f"POINTS {npoints} double\n")

        for j in range(ny):
            for i in range(nx):
                file.write(f"{x_edges[i]:.16e} {y_edges[j]:.16e} 0.0\n")
