"""
1D transient heat conduction with internal heat generation
and a convective boundary condition at x = L.

Numerical method:
    - Cell-centered finite volume method
    - Fully implicit Backward Euler time integration
    - Dirichlet boundary condition at x = 0
    - Robin / convective boundary condition at x = L
    - Uniform volumetric heat generation
    - Direct NumPy linear solver

Author: David Batet Romero
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ProblemParameters:
    # Geometry
    L: float = 1.5                 # Wall thickness [m]
    area: float = 1.0              # Cross-sectional area [m²]

    # Material properties
    k: float = 30.0                # Thermal conductivity [W/(m·K)]
    rho: float = 1000.0            # Density [kg/m³]
    cp: float = 10.0               # Specific heat capacity [J/(kg·K)]

    # Boundary and source terms
    T_left: float = 30.0           # Prescribed temperature at x = 0 [°C]
    T_inf: float = 30.0            # External fluid temperature [°C]
    h: float = 1000.0              # Convective heat-transfer coefficient [W/(m²·K)]
    q_vol: float = 1.0e4           # Volumetric heat generation [W/m³]

    # Time parameters
    dt: float = 1.0                # Time step [s]
    t_final: float = 1000.0        # Final simulation time [s]
    T_init: float = 30.0           # Initial temperature [°C]

    # Mesh
    N: int = 150                   # Number of control volumes

    # Convergence
    tolerance: float = 1.0e-6      # Steady-state temporal tolerance [°C]


def build_mesh(params: ProblemParameters):
    """
    Build a uniform cell-centered mesh.

    Returns
    -------
    x_c : ndarray
        Cell-center coordinates.
    dx : float
        Cell size.
    delta_b : float
        Distance from boundary face to adjacent cell center.
    """
    dx = params.L / params.N
    x_c = (np.arange(params.N) + 0.5) * dx
    delta_b = dx / 2.0

    return x_c, dx, delta_b


def analytical_steady_solution(x: np.ndarray, params: ProblemParameters):
    """
    Analytical steady-state solution for 1D heat conduction with
    internal heat generation, Dirichlet condition at x = 0 and
    Robin condition at x = L.

    Governing equation at steady state:
        k d²T/dx² + q_vol = 0
    """
    L = params.L
    k = params.k
    h = params.h
    q_vol = params.q_vol
    T_left = params.T_left
    T_inf = params.T_inf

    C1 = (
        q_vol * L * (1.0 + h * L / (2.0 * k))
        - h * (T_left - T_inf)
    ) / (k + h * L)

    T_exact = -(q_vol / (2.0 * k)) * x**2 + C1 * x + T_left

    return T_exact


def assemble_system(params: ProblemParameters, dx: float, delta_b: float):
    """
    Assemble the finite volume linear system for the implicit formulation.

    The discretized equation is:

        (aP + aT) T_P^{n+1}
        - aW T_W^{n+1}
        - aE T_E^{n+1}
        = Su + aT T_P^n

    Boundary conditions are included through coefficient/source
    modifications following a Patankar-style formulation.
    """
    N = params.N
    k = params.k
    h = params.h
    area = params.area

    # Internal diffusive conductance
    a_int = k * area / dx

    # Boundary conductance: Dirichlet at x = 0
    a_bL = k * area / delta_b

    # Equivalent boundary conductance: conduction from cell center to wall
    # plus convection from wall to external fluid at x = L
    a_bR = k * h * area / (k + h * delta_b)

    # Source term due to uniform volumetric heat generation
    Su = np.full(N, params.q_vol * area * dx, dtype=float)

    # Interior neighbour coefficients
    aW = np.full(N, a_int, dtype=float)
    aE = np.full(N, a_int, dtype=float)

    # Left boundary: Dirichlet condition
    aW[0] = 0.0
    Su[0] += a_bL * params.T_left

    # Right boundary: Robin condition
    aE[-1] = 0.0
    Su[-1] += a_bR * params.T_inf

    # Central coefficient
    aP = aW + aE
    aP[0] += a_bL
    aP[-1] += a_bR

    # Transient coefficient: Backward Euler
    aT = params.rho * params.cp * area * dx / params.dt

    # Global matrix
    A = np.zeros((N, N), dtype=float)

    # Main diagonal
    A[np.arange(N), np.arange(N)] = aP + aT

    # Lower diagonal
    A[np.arange(1, N), np.arange(0, N - 1)] = -aW[1:]

    # Upper diagonal
    A[np.arange(0, N - 1), np.arange(1, N)] = -aE[:-1]

    coefficients = {
        "a_int": a_int,
        "a_bL": a_bL,
        "a_bR": a_bR,
        "aT": aT,
        "aW": aW,
        "aE": aE,
        "aP": aP,
        "Su": Su,
    }

    return A, Su, aT, coefficients


def compute_right_wall_temperature(T_last: float, params: ProblemParameters, delta_b: float):
    """
    Compute the wall temperature at x = L from the Robin boundary condition.

    The wall temperature is obtained by coupling conduction between the last
    cell center and the wall with convection from the wall to the external fluid.
    """
    k = params.k
    h = params.h

    T_wall = (k * T_last + h * delta_b * params.T_inf) / (k + h * delta_b)

    return T_wall


def solve_transient(params: ProblemParameters, live_plot: bool = False):
    """
    Solve the transient heat conduction problem using Backward Euler.

    Parameters
    ----------
    params : ProblemParameters
        Physical, numerical and mesh parameters.
    live_plot : bool
        If True, shows the transient evolution while solving.

    Returns
    -------
    results : dict
        Dictionary containing mesh, numerical solution, analytical solution,
        convergence information and auxiliary quantities.
    """
    x_c, dx, delta_b = build_mesh(params)

    A, Su, aT, coefficients = assemble_system(params, dx, delta_b)

    T = np.full(params.N, params.T_init, dtype=float)
    T_old = T.copy()

    T_steady = analytical_steady_solution(x_c, params)

    nsteps = int(np.ceil(params.t_final / params.dt))

    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 4))
        (line_num,) = ax.plot(x_c, T, label="Numerical solution")
        ax.plot(x_c, T_steady, "--", label="Analytical steady solution")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("T [°C]")
        ax.set_title("1D heat conduction with Robin boundary condition")
        ax.grid(True)
        ax.legend()
        txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
        fig.tight_layout()

    converged = False
    time = 0.0
    temporal_error = np.inf

    for step in range(nsteps):
        time = (step + 1) * params.dt

        # Right-hand side
        b = Su + aT * T_old

        # Solve implicit linear system
        T = np.linalg.solve(A, b)

        # Convergence based on temporal change
        temporal_error = np.max(np.abs(T - T_old))

        if live_plot:
            line_num.set_ydata(T)
            txt.set_text(f"t = {time:.2f} s\nΔT_inf = {temporal_error:.2e} °C")
            fig.canvas.draw_idle()
            plt.pause(0.001)

        if temporal_error < params.tolerance:
            converged = True
            break

        T_old[:] = T

    if live_plot:
        plt.ioff()
        plt.show()

    steady_error = np.max(np.abs(T - T_steady))

    T_wall_left = params.T_left
    T_wall_right = compute_right_wall_temperature(T[-1], params, delta_b)

    results = {
        "x_c": x_c,
        "dx": dx,
        "delta_b": delta_b,
        "T": T,
        "T_steady": T_steady,
        "T_wall_left": T_wall_left,
        "T_wall_right": T_wall_right,
        "time": time,
        "nsteps": step + 1,
        "converged": converged,
        "temporal_error": temporal_error,
        "steady_error": steady_error,
        "coefficients": coefficients,
    }

    return results


def plot_final_solution(results: dict, params: ProblemParameters):
    """
    Plot the final numerical solution and the analytical steady-state solution.
    """
    x_c = results["x_c"]
    T = results["T"]

    # Numerical solution including wall values
    x_plot = np.r_[0.0, x_c, params.L]
    T_plot = np.r_[results["T_wall_left"], T, results["T_wall_right"]]

    # Smooth analytical curve
    x_exact = np.linspace(0.0, params.L, 500)
    T_exact = analytical_steady_solution(x_exact, params)

    plt.figure(figsize=(7, 4))
    plt.plot(x_plot, T_plot, "o", markersize=4, label="Numerical FVM")
    plt.plot(x_exact, T_exact, "-", label="Analytical steady solution")
    plt.xlabel("x [m]")
    plt.ylabel("T [°C]")
    plt.title("1D heat conduction with internal generation and Robin boundary")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_summary(results: dict, params: ProblemParameters):
    """
    Print a concise numerical summary of the simulation.
    """
    alpha = params.k / (params.rho * params.cp)
    Fo = alpha * params.dt / results["dx"]**2

    print("\n=== Simulation summary ===")
    print(f"Number of control volumes      : {params.N}")
    print(f"dx                             : {results['dx']:.6e} m")
    print(f"dt                             : {params.dt:.6e} s")
    print(f"Thermal diffusivity alpha      : {alpha:.6e} m²/s")
    print(f"Fourier number                 : {Fo:.6e}")
    print(f"Final simulated time           : {results['time']:.6f} s")
    print(f"Number of time steps           : {results['nsteps']}")
    print(f"Converged                      : {results['converged']}")
    print(f"Temporal change L_inf          : {results['temporal_error']:.6e} °C")
    print(f"Steady analytical error L_inf  : {results['steady_error']:.6e} °C")
    print(f"Maximum numerical temperature  : {np.max(results['T']):.6f} °C")
    print(f"Right wall temperature         : {results['T_wall_right']:.6f} °C")


def main():
    params = ProblemParameters(
        L=1.5,
        area=1.0,
        k=30.0,
        rho=1000.0,
        cp=10.0,
        h=1000.0,
        T_left=30.0,
        T_inf=30.0,
        q_vol=1.0e4,
        dt=1.0,
        t_final=1000.0,
        T_init=30.0,
        N=150,
        tolerance=1.0e-6,
    )

    results = solve_transient(params, live_plot=False)

    print_summary(results, params)
    plot_final_solution(results, params)


if __name__ == "__main__":
    main()
