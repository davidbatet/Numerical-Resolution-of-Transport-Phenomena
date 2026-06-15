"""
Finite volume functions for the 2D convection-diffusion problem with ghost cells.

The implementation follows the numerical procedure developed in the thesis:
    - Cell-centered finite volume discretization
    - Ghost-cell boundary treatment
    - Upwind and power-law convection-diffusion schemes
    - Fully implicit transient term
    - Dense matrix assembly
    - Precomputed matrix inverse reused at each time step

Author: David Batet Romero
"""

import numpy as np


def build_mesh_ghost(Lx, Ly, Nx, Ny):
    """
    Build a uniform structured Cartesian mesh for a ghost-cell formulation.
    """
    x_e = np.linspace(0.0, Lx, Nx + 1)
    y_e = np.linspace(0.0, Ly, Ny + 1)

    dx_cell = x_e[1:] - x_e[:-1]
    dy_cell = y_e[1:] - y_e[:-1]

    x_c = 0.5 * (x_e[1:] + x_e[:-1])
    y_c = 0.5 * (y_e[1:] + y_e[:-1])

    Xc, Yc = np.meshgrid(x_c, y_c, indexing="xy")

    return {
        "x_e": x_e,
        "y_e": y_e,
        "x_c": x_c,
        "y_c": y_c,
        "dx_cell": dx_cell,
        "dy_cell": dy_cell,
        "Xc": Xc,
        "Yc": Yc,
    }


def inlet_phi_at_y(y, y_split, phi_in_top, phi_in_bottom):
    """
    Return the inlet scalar value according to the vertical split position.
    """
    if y >= y_split:
        return phi_in_top

    return phi_in_bottom


def fill_ghost_phi(
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
):
    """
    Fill ghost cells according to the boundary conditions.

    Left boundary:
        Dirichlet inlet imposed through ghost-cell symmetry.

    Right boundary:
        Zero-gradient outlet.

    Top and bottom boundaries:
        Zero-gradient / adiabatic condition.
    """
    Ny = len(y_c)
    Nx = len(x_c)

    # Left Dirichlet inlet: phi_G = 2*phi_b - phi_P
    for j in range(1, Ny + 1):
        phi_b = inlet_phi_at_y(
            y_c[j - 1],
            y_split,
            phi_in_top,
            phi_in_bottom,
        )

        phi_g[j, 0] = 2.0 * phi_b - phi_g[j, 1]

    # Right outlet: dphi/dx = 0
    if outlet_zero_grad:
        phi_g[1:Ny + 1, Nx + 1] = phi_g[1:Ny + 1, Nx]

    # Bottom boundary: dphi/dy = 0
    if bottom_adiabatic:
        phi_g[0, :] = phi_g[1, :]

    # Top boundary: dphi/dy = 0
    if top_adiabatic:
        phi_g[Ny + 1, :] = phi_g[Ny, :]


def power_law_A(Pe):
    """
    Patankar power-law interpolation function.
    """
    Pe = abs(Pe)

    return max(0.0, (1.0 - 0.1 * Pe) ** 5)


def assemble_matrix_and_source_ghost(
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
    scheme="upwind",
    top_adiabatic=True,
    bottom_adiabatic=True,
    outlet_zero_grad=True,
):
    """
    Assemble the global matrix and source term for the ghost-cell formulation.

    The original coefficient structure is preserved:
        aP = aE + aW + aN + aS + (Fe - Fw + Fn - Fs) + boundary contribution

    The transient term aP0 is added directly to the matrix diagonal.
    """
    if scheme not in ["upwind", "powerlaw"]:
        raise ValueError("scheme must be either 'upwind' or 'powerlaw'.")

    n_unknowns = Nx * Ny

    K = np.zeros((n_unknowns, n_unknowns), dtype=float)
    Su = np.zeros((Ny, Nx), dtype=float)

    dx_E = np.zeros(Nx, dtype=float)
    dx_W = np.zeros(Nx, dtype=float)

    for i in range(Nx):
        if i < Nx - 1:
            dx_E[i] = x_c[i + 1] - x_c[i]

        if i > 0:
            dx_W[i] = x_c[i] - x_c[i - 1]

    dy_N = np.zeros(Ny, dtype=float)
    dy_S = np.zeros(Ny, dtype=float)

    for j in range(Ny):
        if j < Ny - 1:
            dy_N[j] = y_c[j + 1] - y_c[j]

        if j > 0:
            dy_S[j] = y_c[j] - y_c[j - 1]

    def index(i, j):
        return j * Nx + i

    for j in range(Ny):
        for i in range(Nx):
            P = index(i, j)

            Gamma_P = Gamma_cell[j, i]

            Ae = dy_cell[j]
            Aw = dy_cell[j]
            An = dx_cell[i]
            As = dx_cell[i]

            aE = 0.0
            aW = 0.0
            aN = 0.0
            aS = 0.0

            aP_boundary = 0.0
            SuP = 0.0

            Fe = 0.0
            Fw = 0.0
            Fn = 0.0
            Fs = 0.0

            # East face
            if i < Nx - 1:
                Gamma_E = Gamma_cell[j, i + 1]

                if Gamma_P + Gamma_E != 0.0:
                    Gamma_e = 2.0 * Gamma_P * Gamma_E / (Gamma_P + Gamma_E)
                else:
                    Gamma_e = 0.0

                De = Gamma_e * Ae / dx_E[i]
                Fe = u * Ae

                if scheme == "upwind":
                    aE = De + max(-Fe, 0.0)
                else:
                    Pe = Fe / De if De != 0.0 else 0.0
                    aE = De * power_law_A(Pe) + max(-Fe, 0.0)

                K[P, index(i + 1, j)] = -aE

            else:
                # Outlet boundary: zero diffusive contribution and convective outflow.
                Fe = u * Ae
                aE = 0.0

            # West face
            if i > 0:
                Gamma_W = Gamma_cell[j, i - 1]

                if Gamma_P + Gamma_W != 0.0:
                    Gamma_w = 2.0 * Gamma_P * Gamma_W / (Gamma_P + Gamma_W)
                else:
                    Gamma_w = 0.0

                Dw = Gamma_w * Aw / dx_W[i]
                Fw = u * Aw

                if scheme == "upwind":
                    aW = Dw + max(Fw, 0.0)
                else:
                    Pe = Fw / Dw if Dw != 0.0 else 0.0
                    aW = Dw * power_law_A(Pe) + max(Fw, 0.0)

                K[P, index(i - 1, j)] = -aW

            else:
                # Left inlet Dirichlet condition through ghost-cell formulation.
                Dw_boundary = 2.0 * Gamma_P * Aw / dx_cell[0]
                Fw = u * Aw

                if scheme == "upwind":
                    aW_boundary = Dw_boundary + max(Fw, 0.0)
                else:
                    Pe = Fw / Dw_boundary if Dw_boundary != 0.0 else 0.0
                    aW_boundary = (
                        Dw_boundary * power_law_A(Pe)
                        + max(Fw, 0.0)
                    )

                phi_b = inlet_phi_at_y(
                    y_c[j],
                    y_split,
                    phi_in_top,
                    phi_in_bottom,
                )

                aP_boundary += aW_boundary
                SuP += aW_boundary * phi_b

            # North face
            if j < Ny - 1:
                Gamma_N = Gamma_cell[j + 1, i]

                if Gamma_P + Gamma_N != 0.0:
                    Gamma_n = 2.0 * Gamma_P * Gamma_N / (Gamma_P + Gamma_N)
                else:
                    Gamma_n = 0.0

                Dn = Gamma_n * An / dy_N[j]
                Fn = v * An

                if scheme == "upwind":
                    aN = Dn + max(-Fn, 0.0)
                else:
                    Pe = Fn / Dn if Dn != 0.0 else 0.0
                    aN = Dn * power_law_A(Pe) + max(-Fn, 0.0)

                K[P, index(i, j + 1)] = -aN

            else:
                aN = 0.0
                Fn = v * An

            # South face
            if j > 0:
                Gamma_S = Gamma_cell[j - 1, i]

                if Gamma_P + Gamma_S != 0.0:
                    Gamma_s = 2.0 * Gamma_P * Gamma_S / (Gamma_P + Gamma_S)
                else:
                    Gamma_s = 0.0

                Ds = Gamma_s * As / dy_S[j]
                Fs = v * As

                if scheme == "upwind":
                    aS = Ds + max(Fs, 0.0)
                else:
                    Pe = Fs / Ds if Ds != 0.0 else 0.0
                    aS = Ds * power_law_A(Pe) + max(Fs, 0.0)

                K[P, index(i, j - 1)] = -aS

            else:
                aS = 0.0
                Fs = v * As

            # Conservative Patankar closure
            aP = (
                aE
                + aW
                + aN
                + aS
                + (Fe - Fw + Fn - Fs)
                + aP_boundary
            )

            K[P, P] = aP + aP0[j, i]
            Su[j, i] = SuP

    return K, Su


def build_rhs(Su, aP0, phi_old):
    """
    Build the right-hand-side vector of the implicit system.
    """
    b = Su + aP0 * phi_old

    return b.ravel(order="C")


def invert_matrix(K):
    """
    Compute the inverse of the constant coefficient matrix.
    """
    return np.linalg.inv(K)


def solve_scalar_field(K_inv, Su, aP0, phi_old):
    """
    Solve the implicit scalar transport equation using the precomputed inverse.
    """
    b = build_rhs(Su, aP0, phi_old)
    phi = K_inv @ b

    return phi.reshape(phi_old.shape, order="C")


def write_paraview_structured_grid_vtk(
    x,
    y,
    scalar,
    filename,
    field_name="phi",
):
    """
    Write a 2D scalar field as a VTK legacy STRUCTURED_GRID file.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    scalar = np.asarray(scalar)

    if x.shape != y.shape or x.shape != scalar.shape:
        raise ValueError("Shapes of x, y and scalar must match.")

    ny, nx = x.shape
    npoints = nx * ny

    x_flat = x.ravel(order="C")
    y_flat = y.ravel(order="C")
    scalar_flat = scalar.ravel(order="C")

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

        for value in scalar_flat:
            file.write(f"{value:.16e}\n")
