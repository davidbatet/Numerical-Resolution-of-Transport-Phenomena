"""
Finite volume functions for the Smith-Hutton convection-diffusion problem.

The implementation follows the numerical procedure developed in the thesis:
    - Cell-centered finite volume discretization
    - Patankar-style coefficient formulation
    - Upwind and power-law convection-diffusion schemes
    - Pseudo-transient term for iterative convergence
    - Dense matrix assembly
    - Precomputed matrix inverse reused at each pseudo-time step

Author: David Batet Romero
"""

import numpy as np


def tanh_edges_onesided(ymin, ymax, N, beta):
    """
    Generate a one-sided hyperbolic tangent mesh distribution.

    The refinement is concentrated near ymin. For beta <= 0, a uniform mesh is
    returned.
    """
    if beta is None or beta <= 0.0:
        return np.linspace(ymin, ymax, N + 1)

    length = ymax - ymin
    xi = np.linspace(0.0, 1.0, N + 1)

    eta = 1.0 - np.tanh(beta * (1.0 - xi)) / np.tanh(beta)

    return ymin + length * eta


def build_mesh(
    xmin,
    xmax,
    ymin,
    ymax,
    Nx,
    Ny,
    x_edges=None,
    y_edges=None,
):
    """
    Build a structured Cartesian cell-centered mesh.
    """
    if x_edges is None:
        x_e = np.linspace(xmin, xmax, Nx + 1)
    else:
        x_e = np.asarray(x_edges, dtype=float)
        if len(x_e) != Nx + 1:
            raise ValueError("x_edges must contain Nx + 1 points.")

    if y_edges is None:
        y_e = np.linspace(ymin, ymax, Ny + 1)
    else:
        y_e = np.asarray(y_edges, dtype=float)
        if len(y_e) != Ny + 1:
            raise ValueError("y_edges must contain Ny + 1 points.")

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


def u_field(x, y):
    """
    Horizontal velocity component of the Smith-Hutton velocity field.
    """
    return 2.0 * y * (1.0 - x * x)


def v_field(x, y):
    """
    Vertical velocity component of the Smith-Hutton velocity field.
    """
    return -2.0 * x * (1.0 - y * y)


def power_law_A(Pe):
    """
    Patankar power-law interpolation function.
    """
    Pe = abs(Pe)

    return max(0.0, (1.0 - 0.1 * Pe) ** 5)


def assemble_matrix_and_source_smith_hutton(
    Nx,
    Ny,
    x_e,
    y_e,
    x_c,
    y_c,
    dx_cell,
    dy_cell,
    Gamma,
    rho,
    alpha,
    aP0,
    scheme="upwind",
):
    """
    Assemble the global matrix and source term for the Smith-Hutton problem.

    Boundary treatment:
        - West, east and north boundaries use the constant value
          phi = 1 - tanh(alpha).
        - South boundary for x < 0 uses the Smith-Hutton inlet profile.
        - South boundary for x >= 0 is treated as an outlet condition.

    The original coefficient structure is preserved:

        aP = aE + aW + aN + aS + (Fe - Fw + Fn - Fs) + boundary contribution

    The pseudo-transient term aP0 is added directly to the matrix diagonal.
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

    phi_else = 1.0 - np.tanh(alpha)

    def index(i, j):
        return j * Nx + i

    for j in range(Ny):
        for i in range(Nx):
            P = index(i, j)

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
                De = Gamma * Ae / dx_E[i]
                Fe = rho * u_field(x_e[i + 1], y_c[j]) * Ae

                if scheme == "upwind":
                    aE = De + max(-Fe, 0.0)
                else:
                    aE = De * power_law_A(Fe / De) + max(-Fe, 0.0)

                K[P, index(i + 1, j)] = -aE

            else:
                De = Gamma * Ae / (dx_cell[-1] / 2.0)
                Fe = rho * u_field(x_e[-1], y_c[j]) * Ae

                aE_boundary = De + max(-Fe, 0.0)

                aP_boundary += aE_boundary
                SuP += aE_boundary * phi_else

            # West face
            if i > 0:
                Dw = Gamma * Aw / dx_W[i]
                Fw = rho * u_field(x_e[i], y_c[j]) * Aw

                if scheme == "upwind":
                    aW = Dw + max(Fw, 0.0)
                else:
                    aW = Dw * power_law_A(Fw / Dw) + max(Fw, 0.0)

                K[P, index(i - 1, j)] = -aW

            else:
                Dw = Gamma * Aw / (dx_cell[0] / 2.0)
                Fw = rho * u_field(x_e[0], y_c[j]) * Aw

                aW_boundary = Dw + max(Fw, 0.0)

                aP_boundary += aW_boundary
                SuP += aW_boundary * phi_else

            # North face
            if j < Ny - 1:
                Dn = Gamma * An / dy_N[j]
                Fn = rho * v_field(x_c[i], y_e[j + 1]) * An

                if scheme == "upwind":
                    aN = Dn + max(-Fn, 0.0)
                else:
                    aN = Dn * power_law_A(Fn / Dn) + max(-Fn, 0.0)

                K[P, index(i, j + 1)] = -aN

            else:
                Dn = Gamma * An / (dy_cell[-1] / 2.0)
                Fn = rho * v_field(x_c[i], y_e[-1]) * An

                aN_boundary = Dn + max(-Fn, 0.0)

                aP_boundary += aN_boundary
                SuP += aN_boundary * phi_else

            # South face
            if j > 0:
                Ds = Gamma * As / dy_S[j]
                Fs = rho * v_field(x_c[i], y_e[j]) * As

                if scheme == "upwind":
                    aS = Ds + max(Fs, 0.0)
                else:
                    aS = Ds * power_law_A(Fs / Ds) + max(Fs, 0.0)

                K[P, index(i, j - 1)] = -aS

            else:
                Fs = rho * v_field(x_c[i], y_e[0]) * As

                if x_c[i] < 0.0:
                    phi_boundary = 1.0 + np.tanh(alpha * (2.0 * x_c[i] + 1.0))

                    Ds = Gamma * As / (dy_cell[0] / 2.0)
                    aS_boundary = Ds + max(Fs, 0.0)

                    aP_boundary += aS_boundary
                    SuP += aS_boundary * phi_boundary

                else:
                    aS = max(Fs, 0.0)

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
    Build the right-hand-side vector of the pseudo-transient system.
    """
    b = Su + aP0 * phi_old

    return b.ravel(order="C")


def invert_matrix(K):
    """
    Compute the inverse of the constant coefficient matrix.
    """
    return np.linalg.inv(K)


def solve_interior_field(K_inv, Su, aP0, phi_old):
    """
    Solve the pseudo-transient scalar field using the precomputed inverse matrix.
    """
    b = build_rhs(Su, aP0, phi_old)
    phi_vec = K_inv @ b

    return phi_vec.reshape(phi_old.shape, order="C")


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
                f"0.0\n"
            )

        file.write(f"\nPOINT_DATA {npoints}\n")
        file.write(f"SCALARS {field_name} double 1\n")
        file.write("LOOKUP_TABLE default\n")

        for value in scalar_flat:
            file.write(f"{value:.16e}\n")
