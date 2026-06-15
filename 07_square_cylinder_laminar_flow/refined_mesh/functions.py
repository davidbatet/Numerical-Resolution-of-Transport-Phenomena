"""
Finite volume functions for laminar flow around a square cylinder
on a refined non-uniform MAC grid.

The implementation follows the numerical procedure developed in the thesis:
    - Incompressible Navier-Stokes equations
    - Staggered MAC grid
    - Non-uniform tanh-refined channel mesh
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


def _tanh_cluster_faces(a, b, N, x_ref, beta):
    """
    Build non-uniform faces in [a, b] clustered around x_ref using a
    piecewise symmetric tanh mapping.
    """
    s = np.linspace(0.0, 1.0, N + 1)

    xc = (x_ref - a) / (b - a)
    xc = np.clip(xc, 1.0e-6, 1.0 - 1.0e-6)

    x = np.empty_like(s)

    left = s <= xc
    right = ~left

    if xc > 0.0:
        eta_left = s[left] / xc
        x[left] = a + (x_ref - a) * (
            1.0 + np.tanh(beta * (eta_left - 1.0)) / np.tanh(beta)
        )
    else:
        x[left] = a

    if xc < 1.0:
        eta_right = (s[right] - xc) / (1.0 - xc)
        x[right] = x_ref + (b - x_ref) * (
            np.tanh(beta * eta_right) / np.tanh(beta)
        )
    else:
        x[right] = b

    x[0] = a
    x[-1] = b

    return x


def build_mac_grid_rect_tanh(
    xmin,
    xmax,
    ymin,
    ymax,
    Nx,
    Ny,
    x_refine,
    y_refine,
    beta_x=2.5,
    beta_y=2.5,
):
    """
    Build a rectangular non-uniform MAC grid with tanh clustering.

    Pressure is stored at cell centers.
    u velocity is stored at vertical faces.
    v velocity is stored at horizontal faces.
    """
    x_e = _tanh_cluster_faces(xmin, xmax, Nx, x_refine, beta_x)
    y_e = _tanh_cluster_faces(ymin, ymax, Ny, y_refine, beta_y)

    x_c = 0.5 * (x_e[:-1] + x_e[1:])
    y_c = 0.5 * (y_e[:-1] + y_e[1:])

    dx_cell = x_e[1:] - x_e[:-1]
    dy_cell = y_e[1:] - y_e[:-1]

    x_u = x_e.copy()
    y_u = y_c.copy()

    x_v = x_c.copy()
    y_v = y_e.copy()

    Xc, Yc = np.meshgrid(x_c, y_c, indexing="xy")
    Xu, Yu = np.meshgrid(x_u, y_u, indexing="xy")
    Xv, Yv = np.meshgrid(x_v, y_v, indexing="xy")

    dx_u_cv = np.zeros(Nx + 1)
    dx_u_cv[1:Nx] = x_c[1:] - x_c[:-1]
    dx_u_cv[0] = dx_cell[0]
    dx_u_cv[Nx] = dx_cell[-1]

    dy_u_cv = dy_cell.copy()

    dy_v_cv = np.zeros(Ny + 1)
    dy_v_cv[1:Ny] = y_c[1:] - y_c[:-1]
    dy_v_cv[0] = dy_cell[0]
    dy_v_cv[Ny] = dy_cell[-1]

    dx_v_cv = dx_cell.copy()

    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "Nx": Nx,
        "Ny": Ny,
        "x_e": x_e,
        "y_e": y_e,
        "x_c": x_c,
        "y_c": y_c,
        "dx_cell": dx_cell,
        "dy_cell": dy_cell,
        "x_u": x_u,
        "y_u": y_u,
        "x_v": x_v,
        "y_v": y_v,
        "dx_u_cv": dx_u_cv,
        "dy_u_cv": dy_u_cv,
        "dx_v_cv": dx_v_cv,
        "dy_v_cv": dy_v_cv,
        "Xc": Xc,
        "Yc": Yc,
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


def divergence_uv_to_cell_centers(u, v, grid):
    """
    Compute velocity divergence at pressure cell centers.
    """
    Nx = grid["Nx"]
    Ny = grid["Ny"]

    dx = grid["dx_cell"][None, :]
    dy = grid["dy_cell"][:, None]

    u_w = u[1:Ny + 1, 0:Nx]
    u_e = u[1:Ny + 1, 1:Nx + 1]

    v_s = v[0:Ny, 1:Nx + 1]
    v_n = v[1:Ny + 1, 1:Nx + 1]

    return (u_e - u_w) / dx + (v_n - v_s) / dy


def grad_p_to_u_faces(p, grid):
    """
    Compute pressure gradient at internal u faces.
    """
    Ny, Nx = p.shape

    x_c = grid["x_c"]

    dpdx = np.zeros((Ny + 2, Nx + 1), dtype=p.dtype)

    dist = x_c[1:] - x_c[:-1]
    dpdx[1:Ny + 1, 1:Nx] = (p[:, 1:] - p[:, :-1]) / dist[None, :]

    return dpdx


def grad_p_to_v_faces(p, grid):
    """
    Compute pressure gradient at internal v faces.
    """
    Ny, Nx = p.shape

    y_c = grid["y_c"]

    dpdy = np.zeros((Ny + 1, Nx + 2), dtype=p.dtype)

    dist = y_c[1:] - y_c[:-1]
    dpdy[1:Ny, 1:Nx + 1] = (p[1:, :] - p[:-1, :]) / dist[:, None]

    return dpdy


def correct_velocity(u_pred, v_pred, p, grid, dt, rho=1.0):
    """
    Correct the predicted velocity field using the pressure projection.
    """
    dpdx_u = grad_p_to_u_faces(p, grid)
    dpdy_v = grad_p_to_v_faces(p, grid)

    u_new = u_pred - (dt / rho) * dpdx_u
    v_new = v_pred - (dt / rho) * dpdy_v

    return u_new, v_new


def _laplacian_u(u, grid):
    """
    Compute the non-uniform Laplacian of u at internal u faces.
    """
    Ny_u, Nx1 = u.shape

    Ny = Ny_u - 2
    Nx = Nx1 - 1

    x_u = grid["x_u"]
    y_u = grid["y_u"]

    lap = np.zeros_like(u)

    for j in range(1, Ny + 1):
        for i in range(1, Nx):
            dx_e = x_u[i + 1] - x_u[i]
            dx_w = x_u[i] - x_u[i - 1]
            dx_cv = 0.5 * (dx_e + dx_w)

            if j < Ny:
                dy_n = y_u[j] - y_u[j - 1]
            else:
                dy_n = y_u[j - 1] - y_u[j - 2]

            if j > 1:
                dy_s = y_u[j - 1] - y_u[j - 2]
            else:
                dy_s = y_u[1] - y_u[0]

            dy_cv = 0.5 * (dy_n + dy_s)

            term_x = (
                ((u[j, i + 1] - u[j, i]) / dx_e)
                - ((u[j, i] - u[j, i - 1]) / dx_w)
            ) / dx_cv

            term_y = (
                ((u[j + 1, i] - u[j, i]) / dy_n)
                - ((u[j, i] - u[j - 1, i]) / dy_s)
            ) / dy_cv

            lap[j, i] = term_x + term_y

    return lap


def _laplacian_v(v, grid):
    """
    Compute the non-uniform Laplacian of v at internal v faces.
    """
    Ny1, Nxv = v.shape

    Ny = Ny1 - 1
    Nx = Nxv - 2

    x_v = grid["x_v"]
    y_v = grid["y_v"]

    lap = np.zeros_like(v)

    for j in range(1, Ny):
        for i in range(1, Nx + 1):
            if i < Nx:
                dx_e = x_v[i] - x_v[i - 1]
            else:
                dx_e = x_v[i - 1] - x_v[i - 2]

            if i > 1:
                dx_w = x_v[i - 1] - x_v[i - 2]
            else:
                dx_w = x_v[1] - x_v[0]

            dx_cv = 0.5 * (dx_e + dx_w)

            dy_n = y_v[j + 1] - y_v[j]
            dy_s = y_v[j] - y_v[j - 1]
            dy_cv = 0.5 * (dy_n + dy_s)

            term_x = (
                ((v[j, i + 1] - v[j, i]) / dx_e)
                - ((v[j, i] - v[j, i - 1]) / dx_w)
            ) / dx_cv

            term_y = (
                ((v[j + 1, i] - v[j, i]) / dy_n)
                - ((v[j, i] - v[j - 1, i]) / dy_s)
            ) / dy_cv

            lap[j, i] = term_x + term_y

    return lap


def compute_R_u(u, v, grid, nu, scheme="central"):
    """
    Compute the explicit right-hand side of the u-momentum equation.
    """
    Ny_u, Nx1 = u.shape

    Ny = Ny_u - 2
    Nx = Nx1 - 1

    x_u = grid["x_u"]

    Ru = np.zeros_like(u)

    for j in range(1, Ny + 1):
        for i in range(1, Nx):
            dx_cv = 0.5 * ((x_u[i + 1] - x_u[i]) + (x_u[i] - x_u[i - 1]))
            dy_cv = grid["dy_cell"][j - 1]

            uP = u[j, i]
            uE = u[j, i + 1]
            uW = u[j, i - 1]
            uN = u[j + 1, i]
            uS = u[j - 1, i]

            u_e = 0.5 * (uP + uE)
            u_w = 0.5 * (uW + uP)

            if scheme == "central":
                phi_e = u_e
                phi_w = u_w

            elif scheme == "upwind":
                phi_e = uP if u_e >= 0.0 else uE
                phi_w = uW if u_w >= 0.0 else uP

            else:
                raise ValueError("scheme must be 'central' or 'upwind'.")

            F_uu_e = u_e * phi_e
            F_uu_w = u_w * phi_w

            d_uu_dx = (F_uu_e - F_uu_w) / dx_cv

            v_n = 0.5 * (v[j, i] + v[j, i + 1])
            v_s = 0.5 * (v[j - 1, i] + v[j - 1, i + 1])

            u_n = 0.5 * (uP + uN)
            u_s = 0.5 * (uS + uP)

            if scheme == "central":
                phi_n = u_n
                phi_s = u_s

            else:
                phi_n = uP if v_n >= 0.0 else uN
                phi_s = uS if v_s >= 0.0 else uP

            F_vu_n = v_n * phi_n
            F_vu_s = v_s * phi_s

            d_vu_dy = (F_vu_n - F_vu_s) / dy_cv

            Ru[j, i] = -(d_uu_dx + d_vu_dy)

    lap = _laplacian_u(u, grid)
    Ru[1:Ny + 1, 1:Nx] += nu * lap[1:Ny + 1, 1:Nx]

    return Ru


def compute_R_v(u, v, grid, nu, scheme="central"):
    """
    Compute the explicit right-hand side of the v-momentum equation.
    """
    Ny1, Nxv = v.shape

    Ny = Ny1 - 1
    Nx = Nxv - 2

    y_v = grid["y_v"]

    Rv = np.zeros_like(v)

    for j in range(1, Ny):
        for i in range(1, Nx + 1):
            dx_cv = grid["dx_cell"][i - 1]
            dy_cv = 0.5 * ((y_v[j + 1] - y_v[j]) + (y_v[j] - y_v[j - 1]))

            vP = v[j, i]
            vE = v[j, i + 1]
            vW = v[j, i - 1]
            vN = v[j + 1, i]
            vS = v[j - 1, i]

            v_n = 0.5 * (vP + vN)
            v_s = 0.5 * (vS + vP)

            if scheme == "central":
                phi_n = v_n
                phi_s = v_s

            elif scheme == "upwind":
                phi_n = vP if v_n >= 0.0 else vN
                phi_s = vS if v_s >= 0.0 else vP

            else:
                raise ValueError("scheme must be 'central' or 'upwind'.")

            F_vv_n = v_n * phi_n
            F_vv_s = v_s * phi_s

            d_vv_dy = (F_vv_n - F_vv_s) / dy_cv

            u_e = 0.5 * (u[j, i] + u[j + 1, i])
            u_w = 0.5 * (u[j, i - 1] + u[j + 1, i - 1])

            v_e = 0.5 * (vP + vE)
            v_w = 0.5 * (vW + vP)

            if scheme == "central":
                phi_e = v_e
                phi_w = v_w

            else:
                phi_e = vP if u_e >= 0.0 else vE
                phi_w = vW if u_w >= 0.0 else vP

            F_uv_e = u_e * phi_e
            F_uv_w = u_w * phi_w

            d_uv_dx = (F_uv_e - F_uv_w) / dx_cv

            Rv[j, i] = -(d_uv_dx + d_vv_dy)

    lap = _laplacian_v(v, grid)
    Rv[1:Ny, 1:Nx + 1] += nu * lap[1:Ny, 1:Nx + 1]

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
        - np.sqrt(-7.0 + 12.0 * ar[m2] - 4.0 * ar[m2] * ar[m2])
    )

    return out


def delta_2d_local(x, y, X, Y, hx, hy):
    """
    Two-dimensional regularized delta kernel.
    """
    return (
        (1.0 / hx)
        * peskin_phi_1d((x - X) / hx)
        * (1.0 / hy)
        * peskin_phi_1d((y - Y) / hy)
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


def compute_cell_center_velocity(fields, grid):
    """
    Interpolate staggered velocities to pressure cell centers.
    """
    u = fields["u"]
    v = fields["v"]

    Nx = grid["Nx"]
    Ny = grid["Ny"]

    u_faces = u[1:Ny + 1, :]
    v_faces = v[:, 1:Nx + 1]

    u_center = 0.5 * (u_faces[:, :-1] + u_faces[:, 1:])
    v_center = 0.5 * (v_faces[:-1, :] + v_faces[1:, :])

    return u_center, v_center


def _local_spacing_from_centers(arr, idx):
    """
    Estimate a local spacing around an indexed grid point.
    """
    n = len(arr)

    if n == 1:
        return 1.0

    if idx == 0:
        return arr[1] - arr[0]

    if idx == n - 1:
        return arr[-1] - arr[-2]

    return 0.5 * (arr[idx + 1] - arr[idx - 1])


def interpolate_u_to_lagrangian(u, grid, Xlag, Ylag):
    """
    Interpolate Eulerian u velocity to Lagrangian marker points.
    """
    x_u = grid["x_u"]
    y_u = grid["y_u"]

    u_phys = u[1:-1, :]

    values = np.zeros_like(Xlag, dtype=float)

    for k, (Xk, Yk) in enumerate(zip(Xlag, Ylag)):
        i0 = np.argmin(np.abs(x_u - Xk))
        j0 = np.argmin(np.abs(y_u - Yk))

        hx = _local_spacing_from_centers(x_u, i0)
        hy = _local_spacing_from_centers(y_u, j0)

        i_candidates = np.where(np.abs(x_u - Xk) <= 2.0 * hx)[0]
        j_candidates = np.where(np.abs(y_u - Yk) <= 2.0 * hy)[0]

        accum = 0.0

        for j in j_candidates:
            for i in i_candidates:
                hx_ij = _local_spacing_from_centers(x_u, i)
                hy_ij = _local_spacing_from_centers(y_u, j)

                w = (
                    delta_2d_local(x_u[i], y_u[j], Xk, Yk, hx_ij, hy_ij)
                    * hx_ij
                    * hy_ij
                )

                accum += u_phys[j, i] * w

        values[k] = accum

    return values


def interpolate_v_to_lagrangian(v, grid, Xlag, Ylag):
    """
    Interpolate Eulerian v velocity to Lagrangian marker points.
    """
    x_v = grid["x_v"]
    y_v = grid["y_v"]

    v_phys = v[:, 1:-1]

    values = np.zeros_like(Xlag, dtype=float)

    for k, (Xk, Yk) in enumerate(zip(Xlag, Ylag)):
        i0 = np.argmin(np.abs(x_v - Xk))
        j0 = np.argmin(np.abs(y_v - Yk))

        hx = _local_spacing_from_centers(x_v, i0)
        hy = _local_spacing_from_centers(y_v, j0)

        i_candidates = np.where(np.abs(x_v - Xk) <= 2.0 * hx)[0]
        j_candidates = np.where(np.abs(y_v - Yk) <= 2.0 * hy)[0]

        accum = 0.0

        for j in j_candidates:
            for i in i_candidates:
                hx_ij = _local_spacing_from_centers(x_v, i)
                hy_ij = _local_spacing_from_centers(y_v, j)

                w = (
                    delta_2d_local(x_v[i], y_v[j], Xk, Yk, hx_ij, hy_ij)
                    * hx_ij
                    * hy_ij
                )

                accum += v_phys[j, i] * w

        values[k] = accum

    return values


def spread_fx_to_u_faces(FxLag, grid, Xlag, Ylag, ds):
    """
    Spread Lagrangian x-forces to Eulerian u faces.
    """
    x_u = grid["x_u"]
    y_u = grid["y_u"]

    Ny = grid["Ny"]
    Nx = grid["Nx"]

    fx_u = np.zeros((Ny + 2, Nx + 1), dtype=float)
    fx_u_phys = fx_u[1:-1, :]

    for k, (Xk, Yk) in enumerate(zip(Xlag, Ylag)):
        i0 = np.argmin(np.abs(x_u - Xk))
        j0 = np.argmin(np.abs(y_u - Yk))

        hx = _local_spacing_from_centers(x_u, i0)
        hy = _local_spacing_from_centers(y_u, j0)

        i_candidates = np.where(np.abs(x_u - Xk) <= 2.0 * hx)[0]
        j_candidates = np.where(np.abs(y_u - Yk) <= 2.0 * hy)[0]

        for j in j_candidates:
            for i in i_candidates:
                hx_ij = _local_spacing_from_centers(x_u, i)
                hy_ij = _local_spacing_from_centers(y_u, j)

                w = delta_2d_local(x_u[i], y_u[j], Xk, Yk, hx_ij, hy_ij)

                fx_u_phys[j, i] += FxLag[k] * w * ds

    return fx_u


def spread_fy_to_v_faces(FyLag, grid, Xlag, Ylag, ds):
    """
    Spread Lagrangian y-forces to Eulerian v faces.
    """
    x_v = grid["x_v"]
    y_v = grid["y_v"]

    Ny = grid["Ny"]
    Nx = grid["Nx"]

    fy_v = np.zeros((Ny + 1, Nx + 2), dtype=float)
    fy_v_phys = fy_v[:, 1:-1]

    for k, (Xk, Yk) in enumerate(zip(Xlag, Ylag)):
        i0 = np.argmin(np.abs(x_v - Xk))
        j0 = np.argmin(np.abs(y_v - Yk))

        hx = _local_spacing_from_centers(x_v, i0)
        hy = _local_spacing_from_centers(y_v, j0)

        i_candidates = np.where(np.abs(x_v - Xk) <= 2.0 * hx)[0]
        j_candidates = np.where(np.abs(y_v - Yk) <= 2.0 * hy)[0]

        for j in j_candidates:
            for i in i_candidates:
                hx_ij = _local_spacing_from_centers(x_v, i)
                hy_ij = _local_spacing_from_centers(y_v, j)

                w = delta_2d_local(x_v[i], y_v[j], Xk, Yk, hx_ij, hy_ij)

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
    grid,
    nu,
    dt,
    first_step=False,
    scheme="central",
):
    """
    Compute the AB2 momentum predictor before immersed-boundary forcing.
    """
    Ru = compute_R_u(u, v, grid, nu, scheme=scheme)
    Rv = compute_R_v(u, v, grid, nu, scheme=scheme)

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
    """
    u_pred = u_star0.copy()
    v_pred = v_star0.copy()

    FxLag_total = np.zeros(lag["Nlag"], dtype=float)
    FyLag_total = np.zeros(lag["Nlag"], dtype=float)

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

        FxLag_total += force_data["FxLag"]
        FyLag_total += force_data["FyLag"]

        last_slip = compute_lagrangian_slip(u_pred, v_pred, grid, lag)

        n_used = iteration + 1

        if (tol_slip is not None) and (last_slip["slip_max"] < tol_slip):
            break

    return u_pred, v_pred, FxLag_total, FyLag_total, last_slip, n_used


def assemble_K_poisson_channel_nonuniform(grid):
    """
    Assemble the pressure Poisson matrix on a non-uniform channel grid.

    A single pressure reference is imposed at the top-right cell.
    """
    Nx = grid["Nx"]
    Ny = grid["Ny"]

    x_e = grid["x_e"]
    y_e = grid["y_e"]

    x_c = grid["x_c"]
    y_c = grid["y_c"]

    n_cells = Nx * Ny

    K = np.zeros((n_cells, n_cells), dtype=float)

    def index(i, j):
        return j * Nx + i

    for j in range(Ny):
        for i in range(Nx):
            P = index(i, j)

            if i == Nx - 1 and j == Ny - 1:
                K[P, P] = 1.0
                continue

            dxP = x_e[i + 1] - x_e[i]
            dyP = y_e[j + 1] - y_e[j]

            aP = 0.0

            if i < Nx - 1:
                dxe = x_c[i + 1] - x_c[i]
                aE = 1.0 / (dxP * dxe)

                K[P, index(i + 1, j)] += aE
                aP -= aE

            if i > 0:
                dxw = x_c[i] - x_c[i - 1]
                aW = 1.0 / (dxP * dxw)

                K[P, index(i - 1, j)] += aW
                aP -= aW

            if j < Ny - 1:
                dyn = y_c[j + 1] - y_c[j]
                aN = 1.0 / (dyP * dyn)

                K[P, index(i, j + 1)] += aN
                aP -= aN

            if j > 0:
                dys = y_c[j] - y_c[j - 1]
                aS = 1.0 / (dyP * dys)

                K[P, index(i, j - 1)] += aS
                aP -= aS

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


def compute_force_coefficients_from_total_force(FxLag_total, FyLag_total, lag, rho, Uref, D):
    """
    Compute drag and lift coefficients from accumulated IBM Lagrangian forces.
    """
    Fx_fluid = rho * np.sum(FxLag_total) * lag["ds"]
    Fy_fluid = rho * np.sum(FyLag_total) * lag["ds"]

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

    y_e = grid["y_e"]

    dy = y_e[1:] - y_e[:-1]

    Qin = np.sum(u[1:Ny + 1, 0] * dy)
    Qout = np.sum(u[1:Ny + 1, Nx] * dy)

    return Qin, Qout


def compute_divergence_metrics(u, v, grid):
    """
    Compute global divergence diagnostics.
    """
    div = divergence_uv_to_cell_centers(u, v, grid)

    return {
        "div": div,
        "max": np.max(np.abs(div)),
        "l2": np.sqrt(np.mean(div**2)),
    }


def compute_divergence_metrics_ibm_exterior(u, v, grid, lag, band_cells=1):
    """
    Compute divergence diagnostics excluding a band around the immersed body.
    """
    dx_loc = np.mean(grid["dx_cell"])
    dy_loc = np.mean(grid["dy_cell"])

    Xc = grid["Xc"]
    Yc = grid["Yc"]

    xc = lag["xc"]
    yc = lag["yc"]
    D = lag["D"]

    pad_x = band_cells * dx_loc
    pad_y = band_cells * dy_loc

    xL = xc - 0.5 * D - pad_x
    xR = xc + 0.5 * D + pad_x
    yB = yc - 0.5 * D - pad_y
    yT = yc + 0.5 * D + pad_y

    fluid_mask = ~((Xc >= xL) & (Xc <= xR) & (Yc >= yB) & (Yc <= yT))

    div = divergence_uv_to_cell_centers(u, v, grid)

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


def compute_vorticity_cell_center_nonuniform(u_center, v_center, grid):
    """
    Compute cell-centered vorticity on a non-uniform grid.
    """
    x_c = grid["x_c"]
    y_c = grid["y_c"]

    Ny, Nx = u_center.shape

    omega = np.zeros_like(u_center)

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            dx = x_c[i + 1] - x_c[i - 1]
            dy = y_c[j + 1] - y_c[j - 1]

            dvdx = (v_center[j, i + 1] - v_center[j, i - 1]) / dx
            dudy = (u_center[j + 1, i] - u_center[j - 1, i]) / dy

            omega[j, i] = dvdx - dudy

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

    freqs = np.fft.rfftfreq(y.size, d=dt)
    amps = np.abs(np.fft.rfft(y))

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


def run_solver_square_cylinder_peskin_dense_nonuniform(
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
            grid,
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

        div_pred = divergence_uv_to_cell_centers(u_pred, v_pred, grid)

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
            grid,
            dt,
            rho=rho,
        )

        apply_bc_channel(u, v, p, grid, Umax)

        slip_after_proj = compute_lagrangian_slip(u, v, grid, lag)

        Ru_prev[...] = Ru
        Rv_prev[...] = Rv

        first_step = False

        div_metrics = compute_divergence_metrics(u, v, grid)

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

        u_center, v_center = compute_cell_center_velocity(fields_tmp, grid)

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
