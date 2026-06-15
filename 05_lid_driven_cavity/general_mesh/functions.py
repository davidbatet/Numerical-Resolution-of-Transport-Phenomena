"""
Finite volume functions for the 2D lid-driven cavity flow on a general MAC grid.

The implementation follows the numerical procedure developed in the thesis:
    - Staggered MAC grid
    - Optional two-sided tanh mesh refinement
    - Fractional-step / projection method
    - AB2 explicit predictor for the momentum equations
    - First-order upwind convective fluxes
    - Non-uniform finite-difference operators
    - Dense pressure Poisson matrix with one pressure reference
    - Precomputed matrix inverse reused at each time step
    - Optional Numba acceleration for the momentum operators

Author: David Batet Romero
"""

import numpy as np
from numba import njit


def _tanh_stretch_edges(L, N, beta):
    """
    Generate a two-sided hyperbolic tangent distribution in [0, L].

    For beta close to zero, a uniform mesh is returned.
    """
    if beta <= 1.0e-14:
        return np.linspace(0.0, L, N + 1)

    s = np.linspace(-1.0, 1.0, N + 1)
    xi = 0.5 * (1.0 + np.tanh(beta * s) / np.tanh(beta))

    return L * xi


def build_mac_grid_tanh(
    xmin,
    xmax,
    ymin,
    ymax,
    Nx,
    Ny,
    beta_x=2.0,
    beta_y=2.0,
):
    """
    Build a general MAC grid with optional tanh refinement.

    Pressure is stored at cell centers.
    u velocity is stored at vertical faces.
    v velocity is stored at horizontal faces.
    """
    Lx = xmax - xmin
    Ly = ymax - ymin

    x_e = xmin + _tanh_stretch_edges(Lx, Nx, beta_x)
    y_e = ymin + _tanh_stretch_edges(Ly, Ny, beta_y)

    dx_cell = x_e[1:] - x_e[:-1]
    dy_cell = y_e[1:] - y_e[:-1]

    x_c = 0.5 * (x_e[1:] + x_e[:-1])
    y_c = 0.5 * (y_e[1:] + y_e[:-1])

    Xc, Yc = np.meshgrid(x_c, y_c, indexing="xy")

    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "Nx": Nx,
        "Ny": Ny,
        "beta_x": beta_x,
        "beta_y": beta_y,
        "x_e": x_e,
        "y_e": y_e,
        "x_c": x_c,
        "y_c": y_c,
        "dx_cell": dx_cell,
        "dy_cell": dy_cell,
        "Xc": Xc,
        "Yc": Yc,
    }


def create_fields(Nx, Ny, dtype=float):
    """
    Create pressure and staggered velocity fields.

    Arrays:
        p: pressure at cell centers, shape (Ny, Nx)
        u: x-velocity at vertical faces, including ghost rows in y
        v: y-velocity at horizontal faces, including ghost columns in x
    """
    p = np.zeros((Ny, Nx), dtype=dtype)

    u = np.zeros((Ny + 2, Nx + 1), dtype=dtype)
    v = np.zeros((Ny + 1, Nx + 2), dtype=dtype)

    Ru_prev = np.zeros_like(u)
    Rv_prev = np.zeros_like(v)

    return {
        "p": p,
        "u": u,
        "v": v,
        "Ru_prev": Ru_prev,
        "Rv_prev": Rv_prev,
    }


def apply_bc_lid_driven_cavity(u, v, p, U_lid=1.0):
    """
    Apply lid-driven cavity boundary conditions using ghost values.

    Boundary conditions:
        - No-slip walls on left, right and bottom boundaries
        - Moving lid at the top boundary with u = U_lid
        - v = 0 at top and bottom walls
    """
    Ny_u, Nx1 = u.shape

    Ny = Ny_u - 2
    Nx = Nx1 - 1

    # u velocity: bottom and top walls through ghost rows
    u[0, :] = -u[1, :]
    u[Ny + 1, 1:Nx] = 2.0 * U_lid - u[Ny, 1:Nx]

    # u velocity: left and right walls
    u[:, 0] = 0.0
    u[:, Nx] = 0.0

    # v velocity: bottom and top walls
    v[0, :] = 0.0
    v[Ny, :] = 0.0

    # v velocity: left and right walls through ghost columns
    v[:, 0] = -v[:, 1]
    v[:, Nx + 1] = -v[:, Nx]

    return u, v, p


def divergence_uv_to_cell_centers_nonuniform(u, v, dx_cell, dy_cell):
    """
    Compute velocity divergence at pressure cell centers on a non-uniform grid.
    """
    Ny = u.shape[0] - 2
    Nx = u.shape[1] - 1

    dx = dx_cell.reshape(1, Nx)
    dy = dy_cell.reshape(Ny, 1)

    u_w = u[1:Ny + 1, 0:Nx]
    u_e = u[1:Ny + 1, 1:Nx + 1]

    v_s = v[0:Ny, 1:Nx + 1]
    v_n = v[1:Ny + 1, 1:Nx + 1]

    return (u_e - u_w) / dx + (v_n - v_s) / dy


def grad_p_to_u_faces_nonuniform(p, x_c):
    """
    Compute pressure gradient at u-face locations on a non-uniform grid.
    """
    Ny, Nx = p.shape

    dpdx = np.zeros((Ny + 2, Nx + 1), dtype=p.dtype)

    denominator = x_c[1:] - x_c[:-1]
    dpdx[1:Ny + 1, 1:Nx] = (p[:, 1:] - p[:, :-1]) / denominator.reshape(1, Nx - 1)

    return dpdx


def grad_p_to_v_faces_nonuniform(p, y_c):
    """
    Compute pressure gradient at v-face locations on a non-uniform grid.
    """
    Ny, Nx = p.shape

    dpdy = np.zeros((Ny + 1, Nx + 2), dtype=p.dtype)

    denominator = y_c[1:] - y_c[:-1]
    dpdy[1:Ny, 1:Nx + 1] = (p[1:, :] - p[:-1, :]) / denominator.reshape(Ny - 1, 1)

    return dpdy


def correct_velocity_nonuniform(u_pred, v_pred, p, x_c, y_c, dt):
    """
    Correct the predicted velocity field using the pressure projection.
    """
    dpdx_u = grad_p_to_u_faces_nonuniform(p, x_c)
    dpdy_v = grad_p_to_v_faces_nonuniform(p, y_c)

    u = u_pred - dt * dpdx_u
    v = v_pred - dt * dpdy_v

    return u, v


@njit(cache=True)
def _second_derivative_nonuniform(fm, f0, fp, dm, dp):
    """
    Second derivative on a non-uniform one-dimensional stencil.
    """
    return 2.0 * (((fp - f0) / dp) - ((f0 - fm) / dm)) / (dm + dp)


@njit(cache=True)
def laplacian_u_nonuniform(u, x_e, y_c, ymin, ymax):
    """
    Compute the Laplacian of u at internal u faces on a non-uniform grid.
    """
    Ny = u.shape[0] - 2
    Nx = u.shape[1] - 1

    y_u = np.empty(Ny + 2, dtype=np.float64)

    y_u[1:Ny + 1] = y_c
    y_u[0] = 2.0 * ymin - y_u[1]
    y_u[Ny + 1] = 2.0 * ymax - y_u[Ny]

    lap = np.zeros_like(u)

    for j in range(1, Ny + 1):
        dm_y = y_u[j] - y_u[j - 1]
        dp_y = y_u[j + 1] - y_u[j]

        for i in range(1, Nx):
            dm_x = x_e[i] - x_e[i - 1]
            dp_x = x_e[i + 1] - x_e[i]

            d2x = _second_derivative_nonuniform(
                u[j, i - 1],
                u[j, i],
                u[j, i + 1],
                dm_x,
                dp_x,
            )

            d2y = _second_derivative_nonuniform(
                u[j - 1, i],
                u[j, i],
                u[j + 1, i],
                dm_y,
                dp_y,
            )

            lap[j, i] = d2x + d2y

    return lap


@njit(cache=True)
def laplacian_v_nonuniform(v, x_c, y_e, xmin, xmax):
    """
    Compute the Laplacian of v at internal v faces on a non-uniform grid.
    """
    Ny = v.shape[0] - 1
    Nx = v.shape[1] - 2

    x_v = np.empty(Nx + 2, dtype=np.float64)

    x_v[1:Nx + 1] = x_c
    x_v[0] = 2.0 * xmin - x_v[1]
    x_v[Nx + 1] = 2.0 * xmax - x_v[Nx]

    lap = np.zeros_like(v)

    for j in range(1, Ny):
        dm_y = y_e[j] - y_e[j - 1]
        dp_y = y_e[j + 1] - y_e[j]

        for i in range(1, Nx + 1):
            dm_x = x_v[i] - x_v[i - 1]
            dp_x = x_v[i + 1] - x_v[i]

            d2x = _second_derivative_nonuniform(
                v[j, i - 1],
                v[j, i],
                v[j, i + 1],
                dm_x,
                dp_x,
            )

            d2y = _second_derivative_nonuniform(
                v[j - 1, i],
                v[j, i],
                v[j + 1, i],
                dm_y,
                dp_y,
            )

            lap[j, i] = d2x + d2y

    return lap


@njit(cache=True)
def compute_R_u_nonuniform(u, v, x_e, x_c, y_e, y_c, nu, ymin, ymax):
    """
    Compute the explicit right-hand side of the u-momentum equation.

    Convective terms are discretized using first-order upwind fluxes.
    Diffusive terms are discretized using non-uniform central differences.
    """
    Ny = u.shape[0] - 2
    Nx = u.shape[1] - 1

    Ru = np.zeros_like(u)

    lap = laplacian_u_nonuniform(u, x_e, y_c, ymin, ymax)

    for j in range(1, Ny + 1):
        dy_cv = y_e[j] - y_e[j - 1]

        for i in range(1, Nx):
            dx_e = x_e[i + 1] - x_e[i]
            dx_w = x_e[i] - x_e[i - 1]

            uP = u[j, i]
            uE = u[j, i + 1]
            uW = u[j, i - 1]

            u_adv_e = 0.5 * (uP + uE)
            u_adv_w = 0.5 * (uW + uP)

            u_up_e = uP if u_adv_e >= 0.0 else uE
            u_up_w = uW if u_adv_w >= 0.0 else uP

            F_uu_e = u_adv_e * u_up_e
            F_uu_w = u_adv_w * u_up_w

            d_uu_dx = (F_uu_e - F_uu_w) / (0.5 * (dx_e + dx_w))

            v_s = 0.5 * (v[j - 1, i] + v[j - 1, i + 1])
            v_n = 0.5 * (v[j, i] + v[j, i + 1])

            uS = u[j - 1, i]
            uN = u[j + 1, i]

            u_up_n = uP if v_n >= 0.0 else uN
            u_up_s = uS if v_s >= 0.0 else uP

            F_vu_n = v_n * u_up_n
            F_vu_s = v_s * u_up_s

            d_vu_dy = (F_vu_n - F_vu_s) / dy_cv

            conv = d_uu_dx + d_vu_dy

            Ru[j, i] = -conv + nu * lap[j, i]

    return Ru


@njit(cache=True)
def compute_R_v_nonuniform(u, v, x_e, x_c, y_e, y_c, nu, xmin, xmax):
    """
    Compute the explicit right-hand side of the v-momentum equation.

    Convective terms are discretized using first-order upwind fluxes.
    Diffusive terms are discretized using non-uniform central differences.
    """
    Ny = v.shape[0] - 1
    Nx = v.shape[1] - 2

    Rv = np.zeros_like(v)

    lap = laplacian_v_nonuniform(v, x_c, y_e, xmin, xmax)

    for j in range(1, Ny):
        dy_n = y_e[j + 1] - y_e[j]
        dy_s = y_e[j] - y_e[j - 1]

        for i in range(1, Nx + 1):
            vP = v[j, i]
            vN = v[j + 1, i]
            vS = v[j - 1, i]

            v_adv_n = 0.5 * (vP + vN)
            v_adv_s = 0.5 * (vS + vP)

            v_up_n = vP if v_adv_n >= 0.0 else vN
            v_up_s = vS if v_adv_s >= 0.0 else vP

            F_vv_n = v_adv_n * v_up_n
            F_vv_s = v_adv_s * v_up_s

            d_vv_dy = (F_vv_n - F_vv_s) / (0.5 * (dy_n + dy_s))

            u_w = 0.5 * (u[j, i - 1] + u[j + 1, i - 1])
            u_e = 0.5 * (u[j, i] + u[j + 1, i])

            vW = v[j, i - 1]
            vE = v[j, i + 1]

            v_up_e = vP if u_e >= 0.0 else vE
            v_up_w = vW if u_w >= 0.0 else vP

            F_uv_e = u_e * v_up_e
            F_uv_w = u_w * v_up_w

            dx_cv = x_e[i] - x_e[i - 1]

            d_uv_dx = (F_uv_e - F_uv_w) / dx_cv

            conv = d_uv_dx + d_vv_dy

            Rv[j, i] = -conv + nu * lap[j, i]

    return Rv


def predictor_AB2_nonuniform(
    u,
    v,
    Ru_prev,
    Rv_prev,
    x_e,
    x_c,
    y_e,
    y_c,
    nu,
    dt,
    xmin,
    xmax,
    ymin,
    ymax,
    first_step=False,
):
    """
    Compute the explicit momentum predictor.

    The first step uses Forward Euler. Subsequent steps use AB2.
    """
    Ru = compute_R_u_nonuniform(
        u,
        v,
        x_e,
        x_c,
        y_e,
        y_c,
        nu,
        ymin,
        ymax,
    )

    Rv = compute_R_v_nonuniform(
        u,
        v,
        x_e,
        x_c,
        y_e,
        y_c,
        nu,
        xmin,
        xmax,
    )

    u_pred = u.copy()
    v_pred = v.copy()

    if first_step:
        u_pred += dt * Ru
        v_pred += dt * Rv
    else:
        u_pred += dt * (1.5 * Ru - 0.5 * Ru_prev)
        v_pred += dt * (1.5 * Rv - 0.5 * Rv_prev)

    return u_pred, v_pred, Ru, Rv


def assemble_K_poisson_neumann_nonuniform(
    x_e,
    x_c,
    y_e,
    y_c,
    pin_reference=True,
    ref_i=0,
    ref_j=0,
):
    """
    Assemble the pressure Poisson matrix on a non-uniform cell-centered grid.

    Homogeneous Neumann boundary conditions are applied on all boundaries.
    Since the Neumann Poisson problem is singular, one pressure value is fixed
    when pin_reference=True.
    """
    Nx = len(x_c)
    Ny = len(y_c)

    n_cells = Nx * Ny

    K = np.zeros((n_cells, n_cells), dtype=float)

    dx_cell = x_e[1:] - x_e[:-1]
    dy_cell = y_e[1:] - y_e[:-1]

    def index(i, j):
        return j * Nx + i

    for j in range(Ny):
        dyP = dy_cell[j]

        dy_n = y_c[j + 1] - y_c[j] if j < Ny - 1 else None
        dy_s = y_c[j] - y_c[j - 1] if j > 0 else None

        for i in range(Nx):
            dxP = dx_cell[i]

            dx_e = x_c[i + 1] - x_c[i] if i < Nx - 1 else None
            dx_w = x_c[i] - x_c[i - 1] if i > 0 else None

            P = index(i, j)

            aP = 0.0

            if i == 0:
                aE = 1.0 / (dx_e * dxP)

                K[P, index(i + 1, j)] += aE
                aP -= aE

            elif i == Nx - 1:
                aW = 1.0 / (dx_w * dxP)

                K[P, index(i - 1, j)] += aW
                aP -= aW

            else:
                aE = 1.0 / (dx_e * dxP)
                aW = 1.0 / (dx_w * dxP)

                K[P, index(i + 1, j)] += aE
                K[P, index(i - 1, j)] += aW

                aP -= aE + aW

            if j == 0:
                aN = 1.0 / (dy_n * dyP)

                K[P, index(i, j + 1)] += aN
                aP -= aN

            elif j == Ny - 1:
                aS = 1.0 / (dy_s * dyP)

                K[P, index(i, j - 1)] += aS
                aP -= aS

            else:
                aN = 1.0 / (dy_n * dyP)
                aS = 1.0 / (dy_s * dyP)

                K[P, index(i, j + 1)] += aN
                K[P, index(i, j - 1)] += aS

                aP -= aN + aS

            K[P, P] += aP

    if pin_reference:
        P0 = index(int(ref_i), int(ref_j))

        K[P0, :] = 0.0
        K[P0, P0] = 1.0

    return K


def invert_K(K):
    """
    Compute the inverse of the constant pressure Poisson matrix.
    """
    return np.linalg.inv(K)


def build_rhs_from_divergence(
    div,
    dt,
    pin_reference=True,
    ref_i=0,
    ref_j=0,
):
    """
    Build the right-hand side of the pressure Poisson equation.
    """
    b = (div / dt).ravel(order="C").astype(float)

    if pin_reference:
        Nx = div.shape[1]
        k0 = int(ref_j) * Nx + int(ref_i)
        b[k0] = 0.0

    return b


def solve_poisson(K_inv, b, Nx, Ny):
    """
    Solve the pressure Poisson equation using the precomputed inverse matrix.
    """
    p_vec = K_inv @ b

    return p_vec.reshape((Ny, Nx), order="C")


def run_solver_lid_driven_cavity(
    fields,
    grid,
    Re,
    dt,
    nsteps,
    K_inv,
    U_lid=1.0,
    tol_du=1.0e-8,
    tol_div=1.0e-8,
    tol_dp=1.0e-8,
    verbose_every=50,
):
    """
    Run the lid-driven cavity solver using a projection method.
    """
    Nx = grid["Nx"]
    Ny = grid["Ny"]

    nu = 1.0 / Re

    x_e = grid["x_e"]
    y_e = grid["y_e"]

    x_c = grid["x_c"]
    y_c = grid["y_c"]

    dx_cell = grid["dx_cell"]
    dy_cell = grid["dy_cell"]

    xmin = grid["xmin"]
    xmax = grid["xmax"]
    ymin = grid["ymin"]
    ymax = grid["ymax"]

    u = fields["u"]
    v = fields["v"]
    p = fields["p"]

    Ru_prev = fields["Ru_prev"]
    Rv_prev = fields["Rv_prev"]

    first_step = True

    for step in range(nsteps):
        u_old = u.copy()
        v_old = v.copy()
        p_old = p.copy()

        apply_bc_lid_driven_cavity(u, v, p, U_lid=U_lid)

        u_pred, v_pred, Ru, Rv = predictor_AB2_nonuniform(
            u,
            v,
            Ru_prev,
            Rv_prev,
            x_e,
            x_c,
            y_e,
            y_c,
            nu,
            dt,
            xmin,
            xmax,
            ymin,
            ymax,
            first_step=first_step,
        )

        apply_bc_lid_driven_cavity(u_pred, v_pred, p, U_lid=U_lid)

        div = divergence_uv_to_cell_centers_nonuniform(
            u_pred,
            v_pred,
            dx_cell,
            dy_cell,
        )

        b = build_rhs_from_divergence(
            div,
            dt,
            pin_reference=True,
            ref_i=0,
            ref_j=0,
        )

        p = solve_poisson(K_inv, b, Nx, Ny)

        u, v = correct_velocity_nonuniform(
            u_pred,
            v_pred,
            p,
            x_c,
            y_c,
            dt,
        )

        apply_bc_lid_driven_cavity(u, v, p, U_lid=U_lid)

        Ru_prev[...] = Ru
        Rv_prev[...] = Rv

        first_step = False

        du = np.max(np.abs(u - u_old))
        dv = np.max(np.abs(v - v_old))
        dp = np.max(np.abs(p - p_old))

        max_du = max(du, dv)

        div_corr = divergence_uv_to_cell_centers_nonuniform(
            u,
            v,
            dx_cell,
            dy_cell,
        )

        max_div = np.max(np.abs(div_corr))

        if (step % verbose_every) == 0 or step == nsteps - 1:
            print(
                f"Step {step:6d} | "
                f"max dU = {max_du:.3e} | "
                f"max div = {max_div:.3e} | "
                f"max dP = {dp:.3e}",
                flush=True,
            )

        if max_du < tol_du and max_div < tol_div and dp < tol_dp:
            print(
                f"Converged at step {step} | "
                f"max dU = {max_du:.3e} | "
                f"max div = {max_div:.3e}",
                flush=True,
            )
            break

    fields["u"] = u
    fields["v"] = v
    fields["p"] = p

    return fields


def compute_cell_center_velocity(fields, Nx, Ny):
    """
    Interpolate staggered velocities to pressure cell centers.
    """
    u = fields["u"]
    v = fields["v"]

    u_faces = u[1:Ny + 1, :]
    v_faces = v[:, 1:Nx + 1]

    u_center = 0.5 * (u_faces[:, :-1] + u_faces[:, 1:])
    v_center = 0.5 * (v_faces[:-1, :] + v_faces[1:, :])

    return u_center, v_center


def write_vtk_rectilinear_cell_data(filename, x_e, y_e, cell_data):
    """
    Write scalar and vector cell data to a VTK legacy RECTILINEAR_GRID file.
    """
    x_e = np.asarray(x_e, dtype=float)
    y_e = np.asarray(y_e, dtype=float)

    Nx = x_e.size - 1
    Ny = y_e.size - 1

    def check_field(name, array):
        array = np.asarray(array)

        if array.ndim == 2:
            if array.shape != (Ny, Nx):
                raise ValueError(
                    f"{name} must have shape (Ny, Nx) = ({Ny}, {Nx}), "
                    f"got {array.shape}."
                )

        elif array.ndim == 3:
            if array.shape[:2] != (Ny, Nx) or array.shape[2] not in (2, 3):
                raise ValueError(
                    f"{name} vector field must have shape (Ny, Nx, 2/3), "
                    f"got {array.shape}."
                )

        else:
            raise ValueError(f"{name} has invalid number of dimensions.")

        return array

    def flatten_scalar(array):
        return np.asarray(array, dtype=float).ravel(order="C")

    def flatten_vector(array):
        array = np.asarray(array, dtype=float)

        if array.shape[2] == 2:
            vector = np.zeros((Ny, Nx, 3), dtype=float)
            vector[:, :, 0] = array[:, :, 0]
            vector[:, :, 1] = array[:, :, 1]
        else:
            vector = array

        return vector.reshape((Ny * Nx, 3), order="C")

    with open(filename, "w", encoding="utf-8") as file:
        file.write("# vtk DataFile Version 3.0\n")
        file.write("Lid-driven cavity general mesh\n")
        file.write("ASCII\n")
        file.write("DATASET RECTILINEAR_GRID\n")
        file.write(f"DIMENSIONS {Nx + 1} {Ny + 1} 1\n")

        file.write(f"X_COORDINATES {Nx + 1} float\n")
        file.write(" ".join(f"{value:.16e}" for value in x_e) + "\n")

        file.write(f"Y_COORDINATES {Ny + 1} float\n")
        file.write(" ".join(f"{value:.16e}" for value in y_e) + "\n")

        file.write("Z_COORDINATES 1 float\n")
        file.write("0.0\n")

        n_cells = Nx * Ny
        file.write(f"\nCELL_DATA {n_cells}\n")

        for name, array in cell_data.items():
            array = check_field(name, array)

            if array.ndim == 2:
                flat = flatten_scalar(array)

                file.write(f"SCALARS {name} float 1\n")
                file.write("LOOKUP_TABLE default\n")

                for i in range(0, flat.size, 9):
                    file.write(
                        " ".join(f"{value:.16e}" for value in flat[i:i + 9])
                        + "\n"
                    )

            else:
                flat = flatten_vector(array)

                file.write(f"VECTORS {name} float\n")

                for i in range(flat.shape[0]):
                    file.write(
                        f"{flat[i, 0]:.16e} "
                        f"{flat[i, 1]:.16e} "
                        f"{flat[i, 2]:.16e}\n"
                    )
