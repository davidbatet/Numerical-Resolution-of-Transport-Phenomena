"""
Finite volume functions for laminar flow around a square cylinder
on a uniform MAC grid.

The implementation follows the numerical procedure developed in the thesis:
    - Incompressible Navier-Stokes equations
    - Staggered MAC grid
    - Uniform rectangular channel mesh
    - Parabolic inlet profile and zero-gradient outlet
    - Fractional-step / projection method
    - AB2 explicit predictor for the momentum equations
    - Central or upwind convective discretization
    - Dense pressure Poisson matrix
    - Peskin-type immersed boundary interpolation and spreading
    - Direct forcing on the immersed square cylinder
    - Force-coefficient, Strouhal-number and divergence diagnostics

Author: David Batet Romero
"""

import numpy as np


def build_mac_grid_rect(xmin, xmax, ymin, ymax, Nx, Ny):
    """
    Build a uniform rectangular MAC grid.

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

    x_u = x_e.copy()
    y_u = y_c.copy()

    x_v = x_c.copy()
    y_v = y_e.copy()

    Xu, Yu = np.meshgrid(x_u, y_u, indexing="xy")
    Xv, Yv = np.meshgrid(x_v, y_v, indexing="xy")

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
        "x_u": x_u,
        "y_u": y_u,
        "x_v": x_v,
        "y_v": y_v,
        "Xu": Xu,
        "Yu": Yu,
        "Xv": Xv,
        "Yv": Yv,
    }


def create_fields(Nx, Ny, dtype=float):
    """
    Create pressure and staggered velocity fields.
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


def inlet_parabolic_u(y, ymin, ymax, Umax):
    """
    Parabolic inlet velocity profile.
    """
    H = ymax - ymin
    eta = (y - ymin) / H

    return 4.0 * Umax * eta * (1.0 - eta)


def initialize_channel_flow(fields, grid, Umax):
    """
    Initialize the flow field with the inlet parabolic profile.
    """
    u = fields["u"]
    v = fields["v"]
    p = fields["p"]

    y_c = grid["y_c"]
    u_in = inlet_parabolic_u(y_c, grid["ymin"], grid["ymax"], Umax)

    u[1:-1, :] = u_in[:, None]
    v[:, :] = 0.0
    p[:, :] = 0.0

    apply_bc_channel(u, v, p, grid, Umax)


def add_localized_vertical_perturbation(
    fields,
    grid,
    x0,
    y0,
    D,
    amplitude,
    sigma_x,
    sigma_y,
    antisymmetric_in_y=True,
):
    """
    Add a localized perturbation to the vertical velocity field.
    """
    Xv = grid["Xv"]
    Yv = grid["Yv"]

    gaussian = np.exp(
        -(((Xv - x0) / sigma_x) ** 2 + ((Yv - y0) / sigma_y) ** 2)
    )

    if antisymmetric_in_y:
        shape = ((Yv - y0) / D) * gaussian
    else:
        shape = gaussian

    fields["v"][:, 1:-1] += amplitude * shape


def apply_bc_channel(u, v, p, grid, Umax):
    """
    Apply channel-flow boundary conditions.

    Boundary conditions:
        - No-slip top and bottom walls
        - Parabolic inlet velocity profile
        - Zero-gradient outlet
    """
    Ny_u, Nx1 = u.shape

    Ny = Ny_u - 2
    Nx = Nx1 - 1

    ymin = grid["ymin"]
    ymax = grid["ymax"]
    y_c = grid["y_c"]

    # Top and bottom walls
    u[0, :] = -u[1, :]
    u[Ny + 1, :] = -u[Ny, :]
    v[0, :] = 0.0
    v[Ny, :] = 0.0

    # Inlet
    u_in = inlet_parabolic_u(y_c, ymin, ymax, Umax)

    u[1:Ny + 1, 0] = u_in
    v[:, 1] = 0.0
    v[:, 0] = -v[:, 1]

    # Outlet: zero-gradient approximation
    u[:, Nx] = u[:, Nx - 1]
    v[:, Nx + 1] = v[:, Nx]

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
    Compute pressure gradient at internal u faces.
    """
    Ny, Nx = p.shape

    dpdx = np.zeros((Ny + 2, Nx + 1), dtype=p.dtype)
    dpdx[1:Ny + 1, 1:Nx] = (p[:, 1:] - p[:, :-1]) / dx

    return dpdx


def grad_p_to_v_faces(p, dy):
    """
    Compute pressure gradient at internal v faces.
    """
    Ny, Nx = p.shape

    dpdy = np.zeros((Ny + 1, Nx + 2), dtype=p.dtype)
    dpdy[1:Ny, 1:Nx + 1] = (p[1:, :] - p[:-1, :]) / dy

    return dpdy


def correct_velocity(u_pred, v_pred, p, dx, dy, dt, rho=1.0):
    """
    Correct the predicted velocity field using the pressure projection.
    """
    dpdx_u = grad_p_to_u_faces(p, dx)
    dpdy_v = grad_p_to_v_faces(p, dy)

    u_new = u_pred - (dt / rho) * dpdx_u
    v_new = v_pred - (dt / rho) * dpdy_v

    return u_new, v_new


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


def compute_R_u(u, v, dx, dy, nu, scheme="central"):
    """
    Compute the explicit right-hand side of the u-momentum equation.

    The convective term can be discretized using either central or upwind fluxes.
    Diffusion is discretized using central differences.
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

    if scheme == "central":
        u_e = 0.5 * (uP + uE)
        u_w = 0.5 * (uW + uP)

        F_uu_e = u_e * u_e
        F_uu_w = u_w * u_w

        d_uu_dx = (F_uu_e - F_uu_w) / dx

        v_n = 0.5 * (v[1:Ny + 1, 1:Nx] + v[1:Ny + 1, 2:Nx + 1])
        v_s = 0.5 * (v[0:Ny, 1:Nx] + v[0:Ny, 2:Nx + 1])

        u_n = 0.5 * (uP + uN)
        u_s = 0.5 * (uS + uP)

        F_vu_n = v_n * u_n
        F_vu_s = v_s * u_s

        d_vu_dy = (F_vu_n - F_vu_s) / dy

    elif scheme == "upwind":
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

    else:
        raise ValueError("scheme must be 'central' or 'upwind'.")

    conv = d_uu_dx + d_vu_dy
    lap = _laplacian_u(u, dx, dy)

    Ru[j, i] = -conv + nu * lap[j, i]

    return Ru


def compute_R_v(u, v, dx, dy, nu, scheme="central"):
    """
    Compute the explicit right-hand side of the v-momentum equation.

    The convective term can be discretized using either central or upwind fluxes.
    Diffusion is discretized using central differences.
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

    if scheme == "central":
        v_n = 0.5 * (vP + vN)
        v_s = 0.5 * (vS + vP)

        F_vv_n = v_n * v_n
        F_vv_s = v_s * v_s

        d_vv_dy = (F_vv_n - F_vv_s) / dy

        u_e = 0.5 * (u[1:Ny, 1:Nx + 1] + u[2:Ny + 1, 1:Nx + 1])
        u_w = 0.5 * (u[1:Ny, 0:Nx] + u[2:Ny + 1, 0:Nx])

        v_e = 0.5 * (vP + vE)
        v_w = 0.5 * (vW + vP)

        F_uv_e = u_e * v_e
        F_uv_w = u_w * v_w

        d_uv_dx = (F_uv_e - F_uv_w) / dx

    elif scheme == "upwind":
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

    else:
        raise ValueError("scheme must be 'central' or 'upwind'.")

    conv = d_uv_dx + d_vv_dy
    lap = _laplacian_v(v, dx, dy)

    Rv[j, i] = -conv + nu * lap[j, i]

    return Rv


def peskin_phi_1d(r):
    """
    One-dimensional four-point Peskin kernel.
    """
    ar = np.abs(r)

    out = np.zeros_like(ar)

    m1 = ar < 1.0
    m2 = (ar >= 1.0) & (ar < 2.0)

    out[m1] = (1.0 / 8.0) * (
        3.0
        - 2.0 * ar[m1]
        + np.sqrt(1.0 + 4.0 * ar[m1] - 4.0 * ar[m1] * ar[m1])
    )

    out[m2] = (1.0 / 8.0) * (
        5.0
        - 2.0 * ar[m2]
        - np.sqrt(
            np.clip(
                -7.0 + 12.0 * ar[m2] - 4.0 * ar[m2] * ar[m2],
                0.0,
                None,
            )
        )
    )

    return out


def delta_2d(x, y, X, Y, dx, dy):
    """
    Two-dimensional regularized delta kernel.
    """
    return (
        (1.0 / dx)
        * peskin_phi_1d((x - X) / dx)
        * (1.0 / dy)
        * peskin_phi_1d((y - Y) / dy)
    )


def build_square_lagrangian_boundary(xc, yc, D, ds_target):
    """
    Build Lagrangian marker points on the boundary of a square cylinder.
    """
    n_per_side = max(12, int(np.ceil(D / ds_target)))

    xL = xc - 0.5 * D
    xR = xc + 0.5 * D
    yB = yc - 0.5 * D
    yT = yc + 0.5 * D

    s = np.linspace(0.0, 1.0, n_per_side, endpoint=False)

    xb = xL + D * s
    yb = np.full_like(xb, yB)

    xr = np.full_like(s, xR)
    yr = yB + D * s

    xt = xR - D * s
    yt = np.full_like(xt, yT)

    xl = np.full_like(s, xL)
    yl = yT - D * s

    X = np.concatenate([xb, xr, xt, xl])
    Y = np.concatenate([yb, yr, yt, yl])

    ds = D / n_per_side

    return {
        "X": X.copy(),
        "Y": Y.copy(),
        "ds": ds,
        "Nlag": X.size,
        "xc": xc,
        "yc": yc,
        "D": D,
        "n_per_side": n_per_side,
    }


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


def interpolate_u_to_lagrangian(u, grid, Xlag, Ylag):
    """
    Interpolate Eulerian u velocity to Lagrangian marker points.
    """
    dx = grid["dx"]
    dy = grid["dy"]

    x_u = grid["x_u"]
    y_u = grid["y_u"]

    u_phys = u[1:-1, :]

    values = np.zeros_like(Xlag, dtype=float)

    for k, (Xk, Yk) in enumerate(zip(Xlag, Ylag)):
        i_candidates = np.where(np.abs(x_u - Xk) <= 2.0 * dx)[0]
        j_candidates = np.where(np.abs(y_u - Yk) <= 2.0 * dy)[0]

        accum = 0.0

        for j in j_candidates:
            for i in i_candidates:
                w = delta_2d(x_u[i], y_u[j], Xk, Yk, dx, dy) * dx * dy
                accum += u_phys[j, i] * w

        values[k] = accum

    return values


def interpolate_v_to_lagrangian(v, grid, Xlag, Ylag):
    """
    Interpolate Eulerian v velocity to Lagrangian marker points.
    """
    dx = grid["dx"]
    dy = grid["dy"]

    x_v = grid["x_v"]
    y_v = grid["y_v"]

    v_phys = v[:, 1:-1]

    values = np.zeros_like(Xlag, dtype=float)

    for k, (Xk, Yk) in enumerate(zip(Xlag, Ylag)):
        i_candidates = np.where(np.abs(x_v - Xk) <= 2.0 * dx)[0]
        j_candidates = np.where(np.abs(y_v - Yk) <= 2.0 * dy)[0]

        accum = 0.0

        for j in j_candidates:
            for i in i_candidates:
                w = delta_2d(x_v[i], y_v[j], Xk, Yk, dx, dy) * dx * dy
                accum += v_phys[j, i] * w

        values[k] = accum

    return values


def spread_fx_to_u_faces(FxLag, grid, Xlag, Ylag, ds):
    """
    Spread Lagrangian x-forces to Eulerian u faces.
    """
    dx = grid["dx"]
    dy = grid["dy"]

    x_u = grid["x_u"]
    y_u = grid["y_u"]

    Ny = grid["Ny"]
    Nx = grid["Nx"]

    fx_u = np.zeros((Ny + 2, Nx + 1), dtype=float)
    fx_u_phys = fx_u[1:-1, :]

    for k, (Xk, Yk) in enumerate(zip(Xlag, Ylag)):
        i_candidates = np.where(np.abs(x_u - Xk) <= 2.0 * dx)[0]
        j_candidates = np.where(np.abs(y_u - Yk) <= 2.0 * dy)[0]

        for j in j_candidates:
            for i in i_candidates:
                w = delta_2d(x_u[i], y_u[j], Xk, Yk, dx, dy)
                fx_u_phys[j, i] += FxLag[k] * w * ds

    return fx_u


def spread_fy_to_v_faces(FyLag, grid, Xlag, Ylag, ds):
    """
    Spread Lagrangian y-forces to Eulerian v faces.
    """
    dx = grid["dx"]
    dy = grid["dy"]

    x_v = grid["x_v"]
    y_v = grid["y_v"]

    Ny = grid["Ny"]
    Nx = grid["Nx"]

    fy_v = np.zeros((Ny + 1, Nx + 2), dtype=float)
    fy_v_phys = fy_v[:, 1:-1]

    for k, (Xk, Yk) in enumerate(zip(Xlag, Ylag)):
        i_candidates = np.where(np.abs(x_v - Xk) <= 2.0 * dx)[0]
        j_candidates = np.where(np.abs(y_v - Yk) <= 2.0 * dy)[0]

        for j in j_candidates:
            for i in i_candidates:
                w = delta_2d(x_v[i], y_v[j], Xk, Yk, dx, dy)
                fy_v_phys[j, i] += FyLag[k] * w * ds

    return fy_v


def compute_lagrangian_slip(u_field, v_field, grid, lag):
    """
    Compute velocity slip at the Lagrangian immersed-boundary markers.
    """
    Xlag = lag["X"]
    Ylag = lag["Y"]

    u_lag = interpolate_u_to_lagrangian(u_field, grid, Xlag, Ylag)
    v_lag = interpolate_v_to_lagrangian(v_field, grid, Xlag, Ylag)

    slip = np.sqrt(u_lag**2 + v_lag**2)

    return {
        "u_lag": u_lag,
        "v_lag": v_lag,
        "slip_max": np.max(np.abs(slip)),
        "slip_l2": np.sqrt(np.mean(slip**2)),
    }


def compute_ibm_force_from_velocity(u_in, v_in, grid, lag, dt, force_relax=1.0):
    """
    Compute direct-forcing IBM force from the interpolated Lagrangian velocity.
    """
    Xlag = lag["X"]
    Ylag = lag["Y"]
    ds = lag["ds"]

    u_lag = interpolate_u_to_lagrangian(u_in, grid, Xlag, Ylag)
    v_lag = interpolate_v_to_lagrangian(v_in, grid, Xlag, Ylag)

    FxLag = force_relax * (-u_lag / dt)
    FyLag = force_relax * (-v_lag / dt)

    fx_u = spread_fx_to_u_faces(FxLag, grid, Xlag, Ylag, ds)
    fy_v = spread_fy_to_v_faces(FyLag, grid, Xlag, Ylag, ds)

    slip = np.sqrt(u_lag**2 + v_lag**2)

    force_data = {
        "FxLag": FxLag,
        "FyLag": FyLag,
        "u_lag": u_lag,
        "v_lag": v_lag,
        "slip_lag_max": np.max(np.abs(slip)),
        "slip_lag_l2": np.sqrt(np.mean(slip**2)),
    }

    return fx_u, fy_v, force_data


def predictor_AB2_without_ibm(
    u,
    v,
    Ru_prev,
    Rv_prev,
    dx,
    dy,
    nu,
    dt,
    first_step=False,
    scheme="central",
):
    """
    Compute the AB2 momentum predictor before immersed-boundary forcing.
    """
    Ru = compute_R_u(u, v, dx, dy, nu, scheme=scheme)
    Rv = compute_R_v(u, v, dx, dy, nu, scheme=scheme)

    u_star0 = u.copy()
    v_star0 = v.copy()

    if first_step:
        u_star0 += dt * Ru
        v_star0 += dt * Rv
    else:
        u_star0 += dt * (1.5 * Ru - 0.5 * Ru_prev)
        v_star0 += dt * (1.5 * Rv - 0.5 * Rv_prev)

    return u_star0, v_star0, Ru, Rv


def ibm_iterative_predictor(
    u_star0,
    v_star0,
    p,
    grid,
    lag,
    dt,
    Umax,
    n_ibm_iter=5,
    force_relax=1.0,
    tol_slip=None,
):
    """
    Apply iterative direct forcing before the pressure projection.

    The net IBM force is evaluated from the difference between the velocity
    field before and after the IBM forcing iterations.
    """
    u_pred = u_star0.copy()
    v_pred = v_star0.copy()

    u_pre = u_star0.copy()
    v_pre = v_star0.copy()

    last_slip = compute_lagrangian_slip(u_pred, v_pred, grid, lag)
    n_used = 0

    for iteration in range(n_ibm_iter):
        fx_u, fy_v, force_data = compute_ibm_force_from_velocity(
            u_pred,
            v_pred,
            grid,
            lag,
            dt,
            force_relax=force_relax,
        )

        u_pred += dt * fx_u
        v_pred += dt * fy_v

        apply_bc_channel(u_pred, v_pred, p, grid, Umax)

        last_slip = compute_lagrangian_slip(u_pred, v_pred, grid, lag)

        n_used = iteration + 1

        if (tol_slip is not None) and (last_slip["slip_max"] < tol_slip):
            break

    fx_net_euler = (u_pred - u_pre) / dt
    fy_net_euler = (v_pred - v_pre) / dt

    FxLag_net = interpolate_u_to_lagrangian(
        fx_net_euler,
        grid,
        lag["X"],
        lag["Y"],
    )

    FyLag_net = interpolate_v_to_lagrangian(
        fy_net_euler,
        grid,
        lag["X"],
        lag["Y"],
    )

    return u_pred, v_pred, FxLag_net, FyLag_net, last_slip, n_used


def assemble_K_poisson_channel(Nx, Ny, dx, dy):
    """
    Assemble the pressure Poisson matrix for the channel domain.

    Homogeneous Neumann pressure conditions are used on the domain boundaries.
    A single pressure reference is imposed at the top-right cell.
    """
    n_cells = Nx * Ny

    K = np.zeros((n_cells, n_cells), dtype=float)

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    def index(i, j):
        return j * Nx + i

    for j in range(Ny):
        for i in range(Nx):
            P = index(i, j)

            if i == Nx - 1 and j == Ny - 1:
                K[P, P] = 1.0
                continue

            aP = 0.0

            if i == 0:
                aP -= inv_dx2
                K[P, index(i + 1, j)] += inv_dx2

            elif i == Nx - 1:
                aP -= inv_dx2
                K[P, index(i - 1, j)] += inv_dx2

            else:
                aP -= 2.0 * inv_dx2
                K[P, index(i - 1, j)] += inv_dx2
                K[P, index(i + 1, j)] += inv_dx2

            if j == 0:
                aP -= inv_dy2
                K[P, index(i, j + 1)] += inv_dy2

            elif j == Ny - 1:
                aP -= inv_dy2
                K[P, index(i, j - 1)] += inv_dy2

            else:
                aP -= 2.0 * inv_dy2
                K[P, index(i, j - 1)] += inv_dy2
                K[P, index(i, j + 1)] += inv_dy2

            K[P, P] += aP

    return K


def invert_K(K):
    """
    Compute the inverse of the constant pressure Poisson matrix.
    """
    return np.linalg.inv(K)


def build_rhs_from_divergence_channel(div, dt, rho=1.0):
    """
    Build the right-hand side of the pressure Poisson equation.
    """
    Ny, Nx = div.shape

    b = (rho * div / dt).ravel(order="C").astype(float)

    k_ref = (Ny - 1) * Nx + (Nx - 1)
    b[k_ref] = 0.0

    return b


def solve_poisson_dense(K_inv, b, Nx, Ny):
    """
    Solve the pressure Poisson equation using the precomputed inverse matrix.
    """
    p_vec = K_inv @ b

    return p_vec.reshape((Ny, Nx), order="C")


def compute_force_coefficients_from_total_force(
    FxLag_total,
    FyLag_total,
    lag,
    rho,
    Uref,
    D,
):
    """
    Compute drag and lift coefficients from accumulated IBM Lagrangian forces.

    The original force scaling used in the thesis code is preserved.
    """
    Fx_fluid = np.sum(FxLag_total) * lag["ds"]
    Fy_fluid = np.sum(FyLag_total) * lag["ds"]

    Fx_body = -Fx_fluid
    Fy_body = -Fy_fluid

    qref = 0.5 * rho * Uref * Uref

    Cd = Fx_body / (qref * D)
    Cl = Fy_body / (qref * D)

    return Cd, Cl


def compute_inlet_outlet_flowrates(u, grid):
    """
    Compute inlet and outlet flow rates.
    """
    Ny = grid["Ny"]
    Nx = grid["Nx"]

    dy = grid["dy"]

    Qin = np.sum(u[1:Ny + 1, 0]) * dy
    Qout = np.sum(u[1:Ny + 1, Nx]) * dy

    return Qin, Qout


def compute_divergence_metrics(u, v, dx, dy):
    """
    Compute global divergence diagnostics.
    """
    div = divergence_uv_to_cell_centers(u, v, dx, dy)

    return {
        "div": div,
        "max": np.max(np.abs(div)),
        "l2": np.sqrt(np.mean(div**2)),
    }


def compute_divergence_metrics_ibm_exterior(u, v, grid, lag, band_cells=1):
    """
    Compute divergence diagnostics excluding a band around the immersed body.
    """
    dx = grid["dx"]
    dy = grid["dy"]

    Xc = grid["Xc"]
    Yc = grid["Yc"]

    xc = lag["xc"]
    yc = lag["yc"]
    D = lag["D"]

    pad_x = band_cells * dx
    pad_y = band_cells * dy

    xL = xc - 0.5 * D - pad_x
    xR = xc + 0.5 * D + pad_x
    yB = yc - 0.5 * D - pad_y
    yT = yc + 0.5 * D + pad_y

    fluid_mask = ~((Xc >= xL) & (Xc <= xR) & (Yc >= yB) & (Yc <= yT))

    div = divergence_uv_to_cell_centers(u, v, dx, dy)

    div_masked = div.copy()
    div_masked[~fluid_mask] = 0.0

    return {
        "div": div_masked,
        "max": np.max(np.abs(div_masked[fluid_mask])),
        "l2": np.sqrt(np.mean(div_masked[fluid_mask]**2)),
    }


def sample_cell_center_velocity_at_point(u_center, v_center, grid, xp, yp):
    """
    Interpolate cell-centered velocity at a physical probe point.
    """
    x_c = grid["x_c"]
    y_c = grid["y_c"]

    uxp = np.array([
        np.interp(xp, x_c, u_center[j, :])
        for j in range(len(y_c))
    ])

    vxp = np.array([
        np.interp(xp, x_c, v_center[j, :])
        for j in range(len(y_c))
    ])

    u_val = np.interp(yp, y_c, uxp)
    v_val = np.interp(yp, y_c, vxp)

    return u_val, v_val


def compute_vorticity_cell_center(u_center, v_center, dx, dy):
    """
    Compute cell-centered vorticity on a uniform grid.
    """
    omega = np.zeros_like(u_center)

    omega[1:-1, 1:-1] = (
        (v_center[1:-1, 2:] - v_center[1:-1, :-2]) / (2.0 * dx)
        - (u_center[2:, 1:-1] - u_center[:-2, 1:-1]) / (2.0 * dy)
    )

    return omega


def compute_strouhal_from_signal(t, sig, D, Uref):
    """
    Estimate the Strouhal number from the dominant FFT frequency.
    """
    t = np.asarray(t, dtype=float)
    sig = np.asarray(sig, dtype=float)

    if t.size < 16:
        return np.nan

    dt = np.mean(np.diff(t))

    y = sig - np.mean(sig)

    window = np.hanning(y.size)
    y_windowed = y * window

    freqs = np.fft.rfftfreq(y.size, d=dt)
    amps = np.abs(np.fft.rfft(y_windowed))

    if freqs.size <= 1:
        return np.nan

    amps[0] = 0.0

    k = np.argmax(amps)
    f = freqs[k]

    return f * D / Uref


def compute_signal_window_stats(sig):
    """
    Compute basic statistics over a signal window.
    """
    sig = np.asarray(sig, dtype=float)

    return {
        "mean": np.mean(sig),
        "min": np.min(sig),
        "max": np.max(sig),
        "pp": np.max(sig) - np.min(sig),
        "rms": np.sqrt(np.mean((sig - np.mean(sig)) ** 2)),
    }


def run_solver_square_cylinder_peskin_dense(
    fields,
    grid,
    lag,
    rho,
    nu,
    dt,
    nsteps,
    K_inv,
    Umax,
    Uref,
    D,
    probe_x,
    probe_y,
    scheme="central",
    n_ibm_iter_pre=5,
    force_relax_pre=1.0,
    tol_slip_pre=5.0e-3,
    verbose_every=100,
):
    """
    Run the square-cylinder immersed-boundary solver.
    """
    Nx = grid["Nx"]
    Ny = grid["Ny"]

    dx = grid["dx"]
    dy = grid["dy"]

    u = fields["u"]
    v = fields["v"]
    p = fields["p"]

    Ru_prev = fields["Ru_prev"]
    Rv_prev = fields["Rv_prev"]

    first_step = True

    history = {
        "time": [],
        "Cd": [],
        "Cl": [],
        "u_probe": [],
        "v_probe": [],
        "divmax": [],
        "divl2": [],
        "divext_max": [],
        "divext_l2": [],
        "max_du": [],
        "slip_ibm_pre_max": [],
        "slip_ibm_pre_l2": [],
        "slip_after_proj_max": [],
        "slip_after_proj_l2": [],
        "Qin": [],
        "Qout": [],
        "mass_imbalance": [],
        "ibm_pre_iters": [],
    }

    for step in range(nsteps):
        t = (step + 1) * dt

        u_old = u.copy()
        v_old = v.copy()

        apply_bc_channel(u, v, p, grid, Umax)

        u_star0, v_star0, Ru, Rv = predictor_AB2_without_ibm(
            u,
            v,
            Ru_prev,
            Rv_prev,
            dx,
            dy,
            nu,
            dt,
            first_step=first_step,
            scheme=scheme,
        )

        apply_bc_channel(u_star0, v_star0, p, grid, Umax)

        u_pred, v_pred, FxLag_pre, FyLag_pre, slip_ibm_pre, n_pre = (
            ibm_iterative_predictor(
                u_star0,
                v_star0,
                p,
                grid,
                lag,
                dt,
                Umax,
                n_ibm_iter=n_ibm_iter_pre,
                force_relax=force_relax_pre,
                tol_slip=tol_slip_pre,
            )
        )

        apply_bc_channel(u_pred, v_pred, p, grid, Umax)

        div_pred = divergence_uv_to_cell_centers(u_pred, v_pred, dx, dy)

        b = build_rhs_from_divergence_channel(
            div_pred,
            dt,
            rho=rho,
        )

        p = solve_poisson_dense(K_inv, b, Nx, Ny)

        u, v = correct_velocity(
            u_pred,
            v_pred,
            p,
            dx,
            dy,
            dt,
            rho=rho,
        )

        apply_bc_channel(u, v, p, grid, Umax)

        slip_after_proj = compute_lagrangian_slip(u, v, grid, lag)

        Ru_prev[...] = Ru
        Rv_prev[...] = Rv

        first_step = False

        div_metrics = compute_divergence_metrics(u, v, dx, dy)

        divext_metrics = compute_divergence_metrics_ibm_exterior(
            u,
            v,
            grid,
            lag,
            band_cells=1,
        )

        max_div = div_metrics["max"]
        l2_div = div_metrics["l2"]

        max_div_ext = divext_metrics["max"]
        l2_div_ext = divext_metrics["l2"]

        max_du = max(
            np.max(np.abs(u - u_old)),
            np.max(np.abs(v - v_old)),
        )

        Qin, Qout = compute_inlet_outlet_flowrates(u, grid)
        mass_imb = np.abs(Qout - Qin) / max(np.abs(Qin), 1.0e-14)

        fields_tmp = {
            "u": u,
            "v": v,
            "p": p,
        }

        u_center, v_center = compute_cell_center_velocity(fields_tmp, Nx, Ny)

        u_probe, v_probe = sample_cell_center_velocity_at_point(
            u_center,
            v_center,
            grid,
            probe_x,
            probe_y,
        )

        Cd, Cl = compute_force_coefficients_from_total_force(
            FxLag_pre,
            FyLag_pre,
            lag,
            rho=rho,
            Uref=Uref,
            D=D,
        )

        history["time"].append(t)
        history["Cd"].append(Cd)
        history["Cl"].append(Cl)
        history["u_probe"].append(u_probe)
        history["v_probe"].append(v_probe)
        history["divmax"].append(max_div)
        history["divl2"].append(l2_div)
        history["divext_max"].append(max_div_ext)
        history["divext_l2"].append(l2_div_ext)
        history["max_du"].append(max_du)
        history["slip_ibm_pre_max"].append(slip_ibm_pre["slip_max"])
        history["slip_ibm_pre_l2"].append(slip_ibm_pre["slip_l2"])
        history["slip_after_proj_max"].append(slip_after_proj["slip_max"])
        history["slip_after_proj_l2"].append(slip_after_proj["slip_l2"])
        history["Qin"].append(Qin)
        history["Qout"].append(Qout)
        history["mass_imbalance"].append(mass_imb)
        history["ibm_pre_iters"].append(n_pre)

        if (step % verbose_every) == 0 or step == nsteps - 1:
            print(
                f"Step {step:6d} | "
                f"t = {t:10.4f} | "
                f"max dU = {max_du:.3e} | "
                f"max div = {max_div:.3e} | "
                f"l2 div = {l2_div:.3e} | "
                f"max div ext = {max_div_ext:.3e} | "
                f"l2 div ext = {l2_div_ext:.3e} | "
                f"slip_pre = {slip_ibm_pre['slip_max']:.3e} | "
                f"slip_proj = {slip_after_proj['slip_max']:.3e} | "
                f"mass_err = {mass_imb:.3e} | "
                f"Cd = {Cd:.5f} | "
                f"Cl = {Cl:.5f} | "
                f"nIBM = {n_pre:d}",
                flush=True,
            )

    fields["u"] = u
    fields["v"] = v
    fields["p"] = p

    return {
        "fields": fields,
        "history": history,
    }
