"""
Finite volume functions for the 2D lid-driven cavity flow on a uniform MAC grid.

The implementation follows the numerical procedure developed in the thesis:
    - Staggered MAC grid
    - Fractional-step / projection method
    - AB2 explicit predictor for the momentum equations
    - First-order upwind convective fluxes
    - Central-difference viscous terms
    - Dense pressure Poisson matrix with one pressure reference
    - Precomputed matrix inverse reused at each time step

Author: David Batet Romero
"""

import numpy as np


def build_mac_grid(xmin, xmax, ymin, ymax, Nx, Ny):
    """
    Build a uniform MAC grid.

    Pressure is stored at cell centers.
    u velocity is stored at vertical faces.
    v velocity is stored at horizontal faces.
    """
    x_e = np.linspace(xmin, xmax, Nx + 1)
    y_e = np.linspace(ymin, ymax, Ny + 1)

    dx = (xmax - xmin) / Nx
    dy = (ymax - ymin) / Ny

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
        "dx": dx,
        "dy": dy,
        "x_e": x_e,
        "y_e": y_e,
        "x_c": x_c,
        "y_c": y_c,
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


def divergence_uv_to_cell_centers(u, v, dx, dy):
    """
    Compute velocity divergence at pressure cell centers.
    """
    Ny = u.shape[0] - 2
    Nx = u.shape[1] - 1

    u_w = u[1:Ny + 1, 0:Nx]
    u_e = u[1:Ny + 1, 1:Nx + 1]

    v_s = v[0:Ny, 1:Nx + 1]
    v_n = v[1:Ny + 1, 1:Nx + 1]

    return (u_e - u_w) / dx + (v_n - v_s) / dy


def grad_p_to_u_faces(p, dx):
    """
    Compute pressure gradient at u-face locations.
    """
    Ny, Nx = p.shape

    dpdx = np.zeros((Ny + 2, Nx + 1), dtype=p.dtype)
    dpdx[1:Ny + 1, 1:Nx] = (p[:, 1:] - p[:, :-1]) / dx

    return dpdx


def grad_p_to_v_faces(p, dy):
    """
    Compute pressure gradient at v-face locations.
    """
    Ny, Nx = p.shape

    dpdy = np.zeros((Ny + 1, Nx + 2), dtype=p.dtype)
    dpdy[1:Ny, 1:Nx + 1] = (p[1:, :] - p[:-1, :]) / dy

    return dpdy


def correct_velocity(u_pred, v_pred, p, dx, dy, dt):
    """
    Correct the predicted velocity field using the pressure projection.
    """
    dpdx_u = grad_p_to_u_faces(p, dx)
    dpdy_v = grad_p_to_v_faces(p, dy)

    u = u_pred - dt * dpdx_u
    v = v_pred - dt * dpdy_v

    return u, v


def _laplacian_u(u, dx, dy):
    """
    Compute the Laplacian of u at internal u faces.
    """
    Ny_u, Nx1 = u.shape

    Ny = Ny_u - 2
    Nx = Nx1 - 1

    lap = np.zeros_like(u)

    j = slice(1, Ny + 1)
    i = slice(1, Nx)

    uP = u[j, i]
    uE = u[j, slice(2, Nx + 1)]
    uW = u[j, slice(0, Nx - 1)]
    uN = u[slice(2, Ny + 2), i]
    uS = u[slice(0, Ny), i]

    lap[j, i] = (
        (uE - 2.0 * uP + uW) / (dx * dx)
        + (uN - 2.0 * uP + uS) / (dy * dy)
    )

    return lap


def _laplacian_v(v, dx, dy):
    """
    Compute the Laplacian of v at internal v faces.
    """
    Ny1, Nxv = v.shape

    Ny = Ny1 - 1
    Nx = Nxv - 2

    lap = np.zeros_like(v)

    j = slice(1, Ny)
    i = slice(1, Nx + 1)

    vP = v[j, i]
    vE = v[j, slice(2, Nx + 2)]
    vW = v[j, slice(0, Nx)]
    vN = v[slice(2, Ny + 1), i]
    vS = v[slice(0, Ny - 1), i]

    lap[j, i] = (
        (vE - 2.0 * vP + vW) / (dx * dx)
        + (vN - 2.0 * vP + vS) / (dy * dy)
    )

    return lap


def compute_R_u(u, v, dx, dy, nu):
    """
    Compute the explicit right-hand side of the u-momentum equation.

    Convective terms are discretized using first-order upwind fluxes.
    Diffusive terms are discretized using central differences.
    """
    Ny_u, Nx1 = u.shape

    Ny = Ny_u - 2
    Nx = Nx1 - 1

    Ru = np.zeros_like(u)

    j = slice(1, Ny + 1)
    i = slice(1, Nx)

    uP = u[j, i]
    uE = u[j, slice(2, Nx + 1)]
    uW = u[j, slice(0, Nx - 1)]
    uN = u[slice(2, Ny + 2), i]
    uS = u[slice(0, Ny), i]

    u_adv_e = 0.5 * (uP + uE)
    u_adv_w = 0.5 * (uW + uP)

    u_up_e = np.where(u_adv_e >= 0.0, uP, uE)
    u_up_w = np.where(u_adv_w >= 0.0, uW, uP)

    F_uu_e = u_adv_e * u_up_e
    F_uu_w = u_adv_w * u_up_w

    d_uu_dx = (F_uu_e - F_uu_w) / dx

    v_n = 0.5 * (v[1:Ny + 1, 1:Nx] + v[1:Ny + 1, 2:Nx + 1])
    v_s = 0.5 * (v[0:Ny, 1:Nx] + v[0:Ny, 2:Nx + 1])

    u_up_n = np.where(v_n >= 0.0, uP, uN)
    u_up_s = np.where(v_s >= 0.0, uS, uP)

    F_vu_n = v_n * u_up_n
    F_vu_s = v_s * u_up_s

    d_vu_dy = (F_vu_n - F_vu_s) / dy

    conv = d_uu_dx + d_vu_dy

    lap = _laplacian_u(u, dx, dy)

    Ru[j, i] = -conv + nu * lap[j, i]

    return Ru


def compute_R_v(u, v, dx, dy, nu):
    """
    Compute the explicit right-hand side of the v-momentum equation.

    Convective terms are discretized using first-order upwind fluxes.
    Diffusive terms are discretized using central differences.
    """
    Ny1, Nxv = v.shape

    Ny = Ny1 - 1
    Nx = Nxv - 2

    Rv = np.zeros_like(v)

    j = slice(1, Ny)
    i = slice(1, Nx + 1)

    vP = v[j, i]
    vE = v[j, slice(2, Nx + 2)]
    vW = v[j, slice(0, Nx)]
    vN = v[slice(2, Ny + 1), i]
    vS = v[slice(0, Ny - 1), i]

    v_adv_n = 0.5 * (vP + vN)
    v_adv_s = 0.5 * (vS + vP)

    v_up_n = np.where(v_adv_n >= 0.0, vP, vN)
    v_up_s = np.where(v_adv_s >= 0.0, vS, vP)

    F_vv_n = v_adv_n * v_up_n
    F_vv_s = v_adv_s * v_up_s

    d_vv_dy = (F_vv_n - F_vv_s) / dy

    u_e = 0.5 * (u[1:Ny, 1:Nx + 1] + u[2:Ny + 1, 1:Nx + 1])
    u_w = 0.5 * (u[1:Ny, 0:Nx] + u[2:Ny + 1, 0:Nx])

    v_up_e = np.where(u_e >= 0.0, vP, vE)
    v_up_w = np.where(u_w >= 0.0, vW, vP)

    F_uv_e = u_e * v_up_e
    F_uv_w = u_w * v_up_w

    d_uv_dx = (F_uv_e - F_uv_w) / dx

    conv = d_uv_dx + d_vv_dy

    lap = _laplacian_v(v, dx, dy)

    Rv[j, i] = -conv + nu * lap[j, i]

    return Rv


def predictor_AB2(
    u,
    v,
    Ru_prev,
    Rv_prev,
    dx,
    dy,
    nu,
    dt,
    first_step=False,
):
    """
    Compute the explicit momentum predictor.

    The first step uses Forward Euler. Subsequent steps use AB2.
    """
    Ru = compute_R_u(u, v, dx, dy, nu)
    Rv = compute_R_v(u, v, dx, dy, nu)

    u_pred = u.copy()
    v_pred = v.copy()

    if first_step:
        u_pred += dt * Ru
        v_pred += dt * Rv
    else:
        u_pred += dt * (1.5 * Ru - 0.5 * Ru_prev)
        v_pred += dt * (1.5 * Rv - 0.5 * Rv_prev)

    return u_pred, v_pred, Ru, Rv


def assemble_K_poisson_neumann(
    Nx,
    Ny,
    dx,
    dy,
    pin_reference=True,
    ref_i=0,
    ref_j=0,
):
    """
    Assemble the pressure Poisson matrix with homogeneous Neumann boundaries.

    Since the Neumann Poisson problem is singular, one pressure value is fixed
    when pin_reference=True.
    """
    n_cells = Nx * Ny

    K = np.zeros((n_cells, n_cells), dtype=float)

    def index(i, j):
        return j * Nx + i

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    for j in range(Ny):
        for i in range(Nx):
            P = index(i, j)

            aP = 0.0

            if i == 0:
                aP -= inv_dx2
                K[P, index(i + 1, j)] += inv_dx2

            elif i == Nx - 1:
                aP -= inv_dx2
                K[P, index(i - 1, j)] += inv_dx2

            else:
                aP -= 2.0 * inv_dx2
                K[P, index(i + 1, j)] += inv_dx2
                K[P, index(i - 1, j)] += inv_dx2

            if j == 0:
                aP -= inv_dy2
                K[P, index(i, j + 1)] += inv_dy2

            elif j == Ny - 1:
                aP -= inv_dy2
                K[P, index(i, j - 1)] += inv_dy2

            else:
                aP -= 2.0 * inv_dy2
                K[P, index(i, j + 1)] += inv_dy2
                K[P, index(i, j - 1)] += inv_dy2

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

    dx = grid["dx"]
    dy = grid["dy"]

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

        u_pred, v_pred, Ru, Rv = predictor_AB2(
            u,
            v,
            Ru_prev,
            Rv_prev,
            dx,
            dy,
            nu,
            dt,
            first_step=first_step,
        )

        apply_bc_lid_driven_cavity(u_pred, v_pred, p, U_lid=U_lid)

        div = divergence_uv_to_cell_centers(u_pred, v_pred, dx, dy)

        b = build_rhs_from_divergence(
            div,
            dt,
            pin_reference=True,
            ref_i=0,
            ref_j=0,
        )

        p = solve_poisson(K_inv, b, Nx, Ny)

        u, v = correct_velocity(u_pred, v_pred, p, dx, dy, dt)

        apply_bc_lid_driven_cavity(u, v, p, U_lid=U_lid)

        Ru_prev[...] = Ru
        Rv_prev[...] = Rv

        first_step = False

        du = np.max(np.abs(u - u_old))
        dv = np.max(np.abs(v - v_old))
        dp = np.max(np.abs(p - p_old))

        max_du = max(du, dv)

        div_corr = divergence_uv_to_cell_centers(u, v, dx, dy)
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
        file.write("Lid-driven cavity uniform mesh\n")
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
