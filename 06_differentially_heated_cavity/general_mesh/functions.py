"""
Finite volume functions for the 2D differentially heated cavity problem
on a general non-uniform MAC grid.

The implementation follows the numerical procedure developed in the thesis:
    - Staggered MAC grid
    - Optional two-sided tanh mesh refinement
    - Boussinesq buoyancy coupling
    - Fractional-step / projection method
    - AB2 explicit predictor for momentum and temperature
    - First-order upwind convective fluxes
    - Non-uniform finite-difference operators
    - Dense pressure Poisson matrix with one pressure reference
    - Precomputed matrix inverse reused at each time step
    - Nusselt-number evaluation and de Vahl Davis-type metrics
    - Optional Numba acceleration for momentum and diffusion operators

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

    Pressure and temperature are stored at cell centers.
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
    Create pressure, temperature and staggered velocity fields.

    Arrays:
        p: pressure at cell centers, shape (Ny, Nx)
        T: temperature at cell centers, shape (Ny, Nx)
        u: x-velocity at vertical faces, including ghost rows in y
        v: y-velocity at horizontal faces, including ghost columns in x
    """
    p = np.zeros((Ny, Nx), dtype=dtype)
    T = np.zeros((Ny, Nx), dtype=dtype)

    u = np.zeros((Ny + 2, Nx + 1), dtype=dtype)
    v = np.zeros((Ny + 1, Nx + 2), dtype=dtype)

    Ru_prev = np.zeros_like(u)
    Rv_prev = np.zeros_like(v)
    RT_prev = np.zeros_like(T)

    return {
        "p": p,
        "T": T,
        "u": u,
        "v": v,
        "Ru_prev": Ru_prev,
        "Rv_prev": Rv_prev,
        "RT_prev": RT_prev,
    }


def apply_bc_velocity_dhc(u, v):
    """
    Apply no-slip boundary conditions for the differentially heated cavity.

    Velocity boundary conditions:
        - u = 0 on vertical walls
        - v = 0 on horizontal walls
        - tangential no-slip values are imposed through ghost cells
    """
    Ny_u, Nx1 = u.shape

    Ny = Ny_u - 2
    Nx = Nx1 - 1

    # u velocity: left and right walls
    u[:, 0] = 0.0
    u[:, Nx] = 0.0

    # u velocity: bottom and top walls through ghost rows
    u[0, :] = -u[1, :]
    u[Ny + 1, :] = -u[Ny, :]

    # v velocity: bottom and top walls
    v[0, :] = 0.0
    v[Ny, :] = 0.0

    # v velocity: left and right walls through ghost columns
    v[:, 0] = -v[:, 1]
    v[:, Nx + 1] = -v[:, Nx]

    return u, v


def build_T_with_ghost(T, Thot=1.0, Tcold=0.0):
    """
    Build a temperature field with ghost cells.

    Thermal boundary conditions:
        - Left wall: prescribed hot temperature
        - Right wall: prescribed cold temperature
        - Bottom and top walls: adiabatic condition
    """
    Ny, Nx = T.shape

    Tg = np.zeros((Ny + 2, Nx + 2), dtype=T.dtype)
    Tg[1:Ny + 1, 1:Nx + 1] = T

    # Dirichlet walls through ghost-cell symmetry
    Tg[1:Ny + 1, 0] = 2.0 * Thot - Tg[1:Ny + 1, 1]
    Tg[1:Ny + 1, Nx + 1] = 2.0 * Tcold - Tg[1:Ny + 1, Nx]

    # Adiabatic bottom and top walls
    Tg[0, :] = Tg[1, :]
    Tg[Ny + 1, :] = Tg[Ny, :]

    # Corner consistency
    Tg[0, 0] = Tg[1, 0]
    Tg[0, Nx + 1] = Tg[1, Nx + 1]
    Tg[Ny + 1, 0] = Tg[Ny, 0]
    Tg[Ny + 1, Nx + 1] = Tg[Ny, Nx + 1]

    return Tg


def get_Tref(Thot, Tcold, mode="cold"):
    """
    Return the buoyancy reference temperature.
    """
    if mode == "cold":
        return Tcold

    if mode == "mean":
        return 0.5 * (Thot + Tcold)

    raise ValueError("mode must be either 'cold' or 'mean'.")


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


def grad_phi_to_u_faces_nonuniform(phi, x_c):
    """
    Compute scalar gradient at u-face locations on a non-uniform grid.
    """
    Ny, Nx = phi.shape

    dpdx = np.zeros((Ny + 2, Nx + 1), dtype=phi.dtype)

    denominator = x_c[1:] - x_c[:-1]
    dpdx[1:Ny + 1, 1:Nx] = (
        (phi[:, 1:] - phi[:, :-1]) / denominator.reshape(1, Nx - 1)
    )

    return dpdx


def grad_phi_to_v_faces_nonuniform(phi, y_c):
    """
    Compute scalar gradient at v-face locations on a non-uniform grid.
    """
    Ny, Nx = phi.shape

    dpdy = np.zeros((Ny + 1, Nx + 2), dtype=phi.dtype)

    denominator = y_c[1:] - y_c[:-1]
    dpdy[1:Ny, 1:Nx + 1] = (
        (phi[1:, :] - phi[:-1, :]) / denominator.reshape(Ny - 1, 1)
    )

    return dpdy


def correct_velocity_nonuniform(u_pred, v_pred, phi, x_c, y_c, dt):
    """
    Correct the predicted velocity field using the pressure projection.
    """
    dpdx_u = grad_phi_to_u_faces_nonuniform(phi, x_c)
    dpdy_v = grad_phi_to_v_faces_nonuniform(phi, y_c)

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
def laplacian_T_nonuniform(Tg, x_c, y_c, x_e, y_e):
    """
    Compute the Laplacian of temperature on a non-uniform cell-centered grid.
    """
    Ny = y_c.size
    Nx = x_c.size

    lap = np.zeros((Ny, Nx), dtype=Tg.dtype)

    for j in range(Ny):
        if j == 0:
            dm_y = y_c[j] - (2.0 * y_e[0] - y_c[j])
        else:
            dm_y = y_c[j] - y_c[j - 1]

        if j == Ny - 1:
            dp_y = (2.0 * y_e[Ny] - y_c[j]) - y_c[j]
        else:
            dp_y = y_c[j + 1] - y_c[j]

        for i in range(Nx):
            if i == 0:
                dm_x = x_c[i] - (2.0 * x_e[0] - x_c[i])
            else:
                dm_x = x_c[i] - x_c[i - 1]

            if i == Nx - 1:
                dp_x = (2.0 * x_e[Nx] - x_c[i]) - x_c[i]
            else:
                dp_x = x_c[i + 1] - x_c[i]

            Tm = Tg[j + 1, i]
            T0 = Tg[j + 1, i + 1]
            Tp = Tg[j + 1, i + 2]

            d2x = _second_derivative_nonuniform(
                Tm,
                T0,
                Tp,
                dm_x,
                dp_x,
            )

            Tm = Tg[j, i + 1]
            T0 = Tg[j + 1, i + 1]
            Tp = Tg[j + 2, i + 1]

            d2y = _second_derivative_nonuniform(
                Tm,
                T0,
                Tp,
                dm_y,
                dp_y,
            )

            lap[j, i] = d2x + d2y

    return lap


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
def compute_R_u_nonuniform(u, v, x_e, x_c, y_e, y_c, Pr, ymin, ymax):
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
            dx_cv = 0.5 * (dx_e + dx_w)

            uP = u[j, i]
            uE = u[j, i + 1]
            uW = u[j, i - 1]

            u_adv_e = 0.5 * (uP + uE)
            u_adv_w = 0.5 * (uW + uP)

            u_up_e = uP if u_adv_e >= 0.0 else uE
            u_up_w = uW if u_adv_w >= 0.0 else uP

            F_uu_e = u_adv_e * u_up_e
            F_uu_w = u_adv_w * u_up_w

            d_uu_dx = (F_uu_e - F_uu_w) / dx_cv

            v_s = 0.5 * (v[j - 1, i] + v[j - 1, i + 1])
            v_n = 0.5 * (v[j, i] + v[j, i + 1])

            uS = u[j - 1, i]
            uN = u[j + 1, i]

            u_up_n = uP if v_n >= 0.0 else uN
            u_up_s = uS if v_s >= 0.0 else uP

            F_vu_n = v_n * u_up_n
            F_vu_s = v_s * u_up_s

            d_vu_dy = (F_vu_n - F_vu_s) / dy_cv

            Ru[j, i] = -(d_uu_dx + d_vu_dy) + Pr * lap[j, i]

    return Ru


@njit(cache=True)
def compute_R_v_nonuniform(
    u,
    v,
    x_e,
    x_c,
    y_e,
    y_c,
    T,
    Ra,
    Pr,
    xmin,
    xmax,
    Tref,
):
    """
    Compute the explicit right-hand side of the v-momentum equation.

    Convective terms are discretized using first-order upwind fluxes.
    Diffusive terms are discretized using non-uniform central differences.
    The buoyancy term is evaluated at v faces.
    """
    Ny = v.shape[0] - 1
    Nx = v.shape[1] - 2

    Rv = np.zeros_like(v)

    lap = laplacian_v_nonuniform(v, x_c, y_e, xmin, xmax)

    for j in range(1, Ny):
        dy_n = y_e[j + 1] - y_e[j]
        dy_s = y_e[j] - y_e[j - 1]
        dy_cv = 0.5 * (dy_n + dy_s)

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

            d_vv_dy = (F_vv_n - F_vv_s) / dy_cv

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

            T_face = 0.5 * (T[j - 1, i - 1] + T[j, i - 1])
            buoyancy = Ra * Pr * (T_face - Tref)

            Rv[j, i] = -(d_uv_dx + d_vv_dy) + Pr * lap[j, i] + buoyancy

    return Rv


def compute_R_T_nonuniform(T, u, v, grid, Thot=1.0, Tcold=0.0):
    """
    Compute the explicit right-hand side of the temperature equation.

    Convective terms are discretized using first-order upwind fluxes.
    Diffusive terms are discretized using non-uniform central differences.
    """
    x_e = grid["x_e"]
    y_e = grid["y_e"]

    x_c = grid["x_c"]
    y_c = grid["y_c"]

    dx_cell = grid["dx_cell"]
    dy_cell = grid["dy_cell"]

    Ny, Nx = T.shape

    Tg = build_T_with_ghost(T, Thot=Thot, Tcold=Tcold)

    u_faces = u[1:Ny + 1, :]
    v_faces = v[:, 1:Nx + 1]

    # x-direction convective fluxes
    T_W = Tg[1:Ny + 1, 0:Nx + 1]
    T_E = Tg[1:Ny + 1, 1:Nx + 2]

    T_up_x = np.where(u_faces >= 0.0, T_W, T_E)
    F_x = u_faces * T_up_x

    dFdx = (F_x[:, 1:] - F_x[:, :-1]) / dx_cell.reshape(1, Nx)

    # y-direction convective fluxes
    T_S = Tg[0:Ny + 1, 1:Nx + 1]
    T_N = Tg[1:Ny + 2, 1:Nx + 1]

    T_up_y = np.where(v_faces >= 0.0, T_S, T_N)
    F_y = v_faces * T_up_y

    dFdy = (F_y[1:, :] - F_y[:-1, :]) / dy_cell.reshape(Ny, 1)

    conv = dFdx + dFdy

    lap = laplacian_T_nonuniform(Tg, x_c, y_c, x_e, y_e)

    return -conv + lap


def predictor_AB2_nonuniform(
    u,
    v,
    T,
    Ru_prev,
    Rv_prev,
    RT_prev,
    grid,
    Ra,
    Pr,
    dt,
    Thot=1.0,
    Tcold=0.0,
    Tref_mode="cold",
    first_step=False,
):
    """
    Compute the explicit AB2 predictor for velocity and temperature.

    The first step uses Forward Euler. Subsequent steps use AB2.
    """
    x_e = grid["x_e"]
    y_e = grid["y_e"]

    x_c = grid["x_c"]
    y_c = grid["y_c"]

    xmin = grid["xmin"]
    xmax = grid["xmax"]
    ymin = grid["ymin"]
    ymax = grid["ymax"]

    Tref = float(get_Tref(Thot, Tcold, mode=Tref_mode))

    Ru = compute_R_u_nonuniform(
        u,
        v,
        x_e,
        x_c,
        y_e,
        y_c,
        Pr,
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
        T,
        Ra,
        Pr,
        xmin,
        xmax,
        Tref,
    )

    RT = compute_R_T_nonuniform(
        T,
        u,
        v,
        grid,
        Thot=Thot,
        Tcold=Tcold,
    )

    u_pred = u.copy()
    v_pred = v.copy()
    T_pred = T.copy()

    if first_step:
        u_pred += dt * Ru
        v_pred += dt * Rv
        T_pred += dt * RT
    else:
        u_pred += dt * (1.5 * Ru - 0.5 * Ru_prev)
        v_pred += dt * (1.5 * Rv - 0.5 * Rv_prev)
        T_pred += dt * (1.5 * RT - 0.5 * RT_prev)

    return u_pred, v_pred, T_pred, Ru, Rv, RT


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
    phi_vec = K_inv @ b

    return phi_vec.reshape((Ny, Nx), order="C")


def run_solver_dhc_nonuniform(
    fields,
    grid,
    Ra,
    Pr,
    Thot,
    Tcold,
    Tref_mode,
    dt,
    nsteps,
    K_inv,
    tol_du=1.0e-6,
    tol_div=1.0e-10,
    tol_dp=1.0e-6,
    tol_dT=1.0e-6,
    verbose_every=500,
):
    """
    Run the differentially heated cavity solver using a projection method.

    The original convergence logic is preserved. The parameter tol_dp is kept
    in the interface for consistency, although the stopping criterion uses
    velocity, divergence and temperature changes.
    """
    Nx = grid["Nx"]
    Ny = grid["Ny"]

    x_c = grid["x_c"]
    y_c = grid["y_c"]

    dx_cell = grid["dx_cell"]
    dy_cell = grid["dy_cell"]

    u = fields["u"]
    v = fields["v"]
    p = fields["p"]
    T = fields["T"]

    Ru_prev = fields["Ru_prev"]
    Rv_prev = fields["Rv_prev"]
    RT_prev = fields["RT_prev"]

    first_step = True

    history = {
        "dt": [],
        "du": [],
        "div": [],
        "dphi": [],
        "dT": [],
        "Nu_hot": [],
    }

    for step in range(nsteps):
        u_old = u.copy()
        v_old = v.copy()
        T_old = T.copy()

        apply_bc_velocity_dhc(u, v)

        u_pred, v_pred, T_pred, Ru, Rv, RT = predictor_AB2_nonuniform(
            u,
            v,
            T,
            Ru_prev,
            Rv_prev,
            RT_prev,
            grid,
            Ra,
            Pr,
            dt,
            Thot=Thot,
            Tcold=Tcold,
            Tref_mode=Tref_mode,
            first_step=first_step,
        )

        apply_bc_velocity_dhc(u_pred, v_pred)

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

        phi = solve_poisson(K_inv, b, Nx, Ny)

        u, v = correct_velocity_nonuniform(
            u_pred,
            v_pred,
            phi,
            x_c,
            y_c,
            dt,
        )

        p += phi
        T = T_pred

        apply_bc_velocity_dhc(u, v)

        Ru_prev[...] = Ru
        Rv_prev[...] = Rv
        RT_prev[...] = RT

        first_step = False

        du = np.max(np.abs(u - u_old))
        dv = np.max(np.abs(v - v_old))
        max_du = max(du, dv)

        dT = np.max(np.abs(T - T_old))

        div_corr = divergence_uv_to_cell_centers_nonuniform(
            u,
            v,
            dx_cell,
            dy_cell,
        )

        max_div = np.max(np.abs(div_corr))

        dpdx_u = grad_phi_to_u_faces_nonuniform(phi, x_c)
        dpdy_v = grad_phi_to_v_faces_nonuniform(phi, y_c)

        dphi = dt * max(
            np.max(np.abs(dpdx_u[1:Ny + 1, 1:Nx])),
            np.max(np.abs(dpdy_v[1:Ny, 1:Nx + 1])),
        )

        Nu_hot_local, Nu_hot_avg, _, _ = compute_nusselt_hot_cold_nonuniform(
            T,
            grid["x_e"],
            x_c,
            Thot=Thot,
            Tcold=Tcold,
        )

        history["dt"].append(dt)
        history["du"].append(max_du)
        history["div"].append(max_div)
        history["dphi"].append(dphi)
        history["dT"].append(dT)
        history["Nu_hot"].append(Nu_hot_avg)

        if (step % verbose_every) == 0 or step == nsteps - 1:
            print(
                f"Step {step:6d} | "
                f"dt = {dt:.2e} | "
                f"max dU = {max_du:.3e} | "
                f"max div = {max_div:.3e} | "
                f"dphi = {dphi:.3e} | "
                f"max dT = {dT:.3e} | "
                f"Nu_hot_avg = {Nu_hot_avg:.6f}",
                flush=True,
            )

        if max_du < tol_du and max_div < tol_div and dT < tol_dT:
            print(
                f"Converged at step {step} | "
                f"max dU = {max_du:.3e} | "
                f"max div = {max_div:.3e} | "
                f"max dT = {dT:.3e} | "
                f"Nu_hot_avg = {Nu_hot_avg:.6f}",
                flush=True,
            )
            break

    fields["u"] = u
    fields["v"] = v
    fields["p"] = p
    fields["T"] = T

    return fields, history


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


def compute_nusselt_hot_cold_nonuniform(T, x_e, x_c, Thot=1.0, Tcold=0.0):
    """
    Compute local and average Nusselt numbers at the hot and cold walls.

    The thermal gradient is evaluated using the first and last cell centers.
    """
    dx_w = x_c[0] - x_e[0]
    dx_e = x_e[-1] - x_c[-1]

    Nu_hot_local = -(T[:, 0] - Thot) / dx_w
    Nu_cold_local = -(Tcold - T[:, -1]) / dx_e

    Nu_hot_avg = float(np.mean(Nu_hot_local))
    Nu_cold_avg = float(np.mean(Nu_cold_local))

    return Nu_hot_local, Nu_hot_avg, Nu_cold_local, Nu_cold_avg


def compute_de_vahl_davis_metrics(fields, grid, Thot=1.0, Tcold=0.0):
    """
    Compute de Vahl Davis-type metrics for natural convection validation.
    """
    Nx = grid["Nx"]
    Ny = grid["Ny"]

    x_c = grid["x_c"]
    y_c = grid["y_c"]
    x_e = grid["x_e"]

    T = fields["T"]

    u_center, v_center = compute_cell_center_velocity(fields, Nx, Ny)

    x_target = 0.5
    u_x05 = np.array([
        np.interp(x_target, x_c, u_center[j, :])
        for j in range(Ny)
    ])

    umax = float(np.max(u_x05))
    y_umax = float(y_c[int(np.argmax(u_x05))])

    y_target = 0.5
    v_y05 = np.array([
        np.interp(y_target, y_c, v_center[:, i])
        for i in range(Nx)
    ])

    vmax = float(np.max(v_y05))
    x_vmax = float(x_c[int(np.argmax(v_y05))])

    Nu_hot_local, Nu_avg, _, _ = compute_nusselt_hot_cold_nonuniform(
        T,
        x_e,
        x_c,
        Thot=Thot,
        Tcold=Tcold,
    )

    # Extrapolate Nu to y = 0
    y0 = float(y_c[0])
    y1 = float(y_c[1])

    Nu0_0 = float(Nu_hot_local[0])
    Nu0_1 = float(Nu_hot_local[1])

    Nu_at_0 = Nu0_0 + (0.0 - y0) * (Nu0_1 - Nu0_0) / (y1 - y0)

    # Extrapolate Nu to y = 1
    yN_1 = float(y_c[-2])
    yN = float(y_c[-1])

    NuN_1 = float(Nu_hot_local[-2])
    NuN = float(Nu_hot_local[-1])

    Nu_at_1 = NuN + (1.0 - yN) * (NuN - NuN_1) / (yN - yN_1)

    Nu_half = float(np.interp(0.5, y_c, Nu_hot_local))

    Nu_max = float(np.max(Nu_hot_local))
    y_Nu_max = float(y_c[int(np.argmax(Nu_hot_local))])

    Nu_min = float(np.min(Nu_hot_local))
    y_Nu_min = float(y_c[int(np.argmin(Nu_hot_local))])

    return {
        "umax": umax,
        "y_umax": y_umax,
        "vmax": vmax,
        "x_vmax": x_vmax,
        "Nu_avg": float(Nu_avg),
        "Nu_0": Nu_at_0,
        "Nu_half": Nu_half,
        "Nu_1": Nu_at_1,
        "Nu_max": Nu_max,
        "y_Nu_max": y_Nu_max,
        "Nu_min": Nu_min,
        "y_Nu_min": y_Nu_min,
    }


def print_de_vahl_davis_summary(metrics, Ra, Pr, Nx, Ny):
    """
    Print a de Vahl Davis-type validation summary.
    """
    print("\n--- de Vahl Davis-like summary ---")
    print(f"Ra = {Ra:g}, Pr = {Pr:g}, Nx = {Nx}, Ny = {Ny}")
    print(f"u_max  = {metrics['umax']:.6f} at y = {metrics['y_umax']:.6f}")
    print(f"v_max  = {metrics['vmax']:.6f} at x = {metrics['x_vmax']:.6f}")
    print(f"Nu_avg = {metrics['Nu_avg']:.6f}")
    print(f"Nu_0   = {metrics['Nu_0']:.6f}   extrapolated at y = 0")
    print(f"Nu_1/2 = {metrics['Nu_half']:.6f} interpolated at y = 0.5")
    print(f"Nu_1   = {metrics['Nu_1']:.6f}   extrapolated at y = 1")
    print(
        f"Nu_max = {metrics['Nu_max']:.6f} "
        f"at y = {metrics['y_Nu_max']:.6f}"
    )
    print(
        f"Nu_min = {metrics['Nu_min']:.6f} "
        f"at y = {metrics['y_Nu_min']:.6f}"
    )


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
        file.write("Differentially heated cavity general mesh\n")
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
