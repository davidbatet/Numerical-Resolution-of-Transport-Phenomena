"""
1D steady heat conduction with internal heat generation
and Dirichlet boundary conditions at both ends.

Numerical method:
    - Cell-centered finite volume method
    - Patankar-style coefficient formulation
    - Uniform mesh
    - Direct NumPy linear solver
    - Analytical steady-state validation

Physical problem:
    k d²T/dx² + q_vol = 0

Boundary conditions:
    T(0) = T_left
    T(L) = T_right

Author: David Batet Romero
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ProblemParameters:
    # Geometry
    L: float = 1.5                 # Domain length [m]
    area: float = 1.0              # Cross-sectional area [m²]

    # Material properties
    k: float = 30.0                # Thermal conductivity [W/(m·K)]

    # Source term
    q_vol: float = 1.0e4           # Volumetric heat generation [W/m³]

    # Boundary conditions
    T_left: float = 30.0           # Prescribed temperature at x = 0 [°C]
    T_right: float = 30.0          # Prescribed temperature at x = L [°C]

    # Mesh
    N: int = 500                   # Number of control volumes


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
    uniform internal heat generation and Dirichlet boundary conditions.

    Governing equation:
        k d²T/dx² + q_vol = 0

    General solution:
        T(x) = -q_vol/(2k) x² + C1 x + T_left

    where:
        C1 = q_vol L/(2k) + (T_right - T_left)/L
    """
    L = params.L
    k = params.k
    q_vol = params.q_vol
    T_left = params.T_left
    T_right = params.T_right

    C1 = q_vol * L / (2.0 * k) + (T_right - T_left) / L

    T_exact = -(q_vol / (2.0 * k)) * x**2 + C1 * x + T_left

    return T_exact


def assemble_system(params: ProblemParameters, dx: float, delta_b: float):
    """
    Assemble the finite volume linear system for the steady formulation.

    The discretized equation is:

        aP T_P - aW T_W - aE T_E = Su

    For the boundary-adjacent cells, Dirichlet boundary conditions are
    included through Patankar-style coefficient/source modifications.
    """
    N = params.N
    k = params.k
    area = params.area

    # Internal diffusive conductance
    a_int = k * area / dx

    # Boundary conductance between cell center and wall
    a_b = k * area / delta_b

    # Source term due to uniform volumetric heat generation
    Su = np.full(N, params.q_vol * area * dx, dtype=float)

    # Interior neighbour coefficients
    aW = np.full(N, a_int, dtype=float)
    aE = np.full(N, a_int, dtype=float)

    # Left boundary: Dirichlet condition
    aW[0] = 0.0
    Su[0] += a_b * params.T_left

    # Right boundary: Dirichlet condition
    aE[-1] = 0.0
    Su[-1] += a_b * params.T_right

    # Central coefficient
    aP = aW + aE
    aP[0] += a_b
    aP[-1] += a_b

    # Global matrix
    A = np.zeros((N, N), dtype=float)

    # Main diagonal
    A[np.arange(N), np.arange(N)] = aP

    # Lower diagonal
    A[np.arange(1, N), np.arange(0, N - 1)] = -aW[1:]

    # Upper diagonal
    A[np.arange(0, N - 1), np.arange(1, N)] = -aE[:-1]

    coefficients = {
        "a_int": a_int,
        "a_b": a_b,
        "aW": aW,
        "aE": aE,
        "aP": aP,
        "Su": Su,
    }

    return A, Su, coefficients


def solve_steady(params: ProblemParameters):
    """
    Solve the steady 1D heat conduction problem.

    Returns
    -------
    results : dict
        Dictionary containing mesh, numerical solution, analytical solution,
        error and auxiliary quantities.
    """
    x_c, dx, delta_b = build_mesh(params)

    A, b, coefficients = assemble_system(params, dx, delta_b)

    # Solve linear system
    T = np.linalg.solve(A, b)

    # Analytical solution evaluated at cell centers
    T_steady = analytical_steady_solution(x_c, params)

    # Error against analytical solution
    steady_error = np.max(np.abs(T - T_steady))

    results = {
        "x_c": x_c,
        "dx": dx,
        "delta_b": delta_b,
        "T": T,
        "T_steady": T_steady,
        "T_wall_left": params.T_left,
        "T_wall_right": params.T_right,
        "steady_error": steady_error,
        "coefficients": coefficients,
    }

    return results


def plot_final_solution(results: dict, params: ProblemParameters):
    """
    Plot the numerical solution and the analytical steady-state solution.
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
    plt.plot(
        x_plot,
        T_plot,
        "o",
        markersize=4,
        markevery=max(params.N // 50, 1),
        label="Numerical FVM",
    )
    plt.plot(x_exact, T_exact, "-", label="Analytical steady solution")
    plt.xlabel("x [m]")
    plt.ylabel("T [°C]")
    plt.title("1D steady heat conduction with internal generation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_summary(results: dict, params: ProblemParameters):
    """
    Print a concise numerical summary of the simulation.
    """
    print("\n=== Simulation summary ===")
    print(f"Number of control volumes      : {params.N}")
    print(f"Domain length                  : {params.L:.6f} m")
    print(f"Cell size                      : {results['dx']:.6e} m")
    print(f"Thermal conductivity           : {params.k:.6f} W/(m·K)")
    print(f"Volumetric heat generation     : {params.q_vol:.6e} W/m³")
    print(f"Left boundary temperature      : {params.T_left:.6f} °C")
    print(f"Right boundary temperature     : {params.T_right:.6f} °C")
    print(f"Maximum numerical temperature  : {np.max(results['T']):.6f} °C")
    print(f"Steady analytical error L_inf  : {results['steady_error']:.6e} °C")


def main():
    params = ProblemParameters(
        L=1.5,
        area=1.0,
        k=30.0,
        q_vol=1.0e4,
        T_left=30.0,
        T_right=30.0,
        N=500,
    )

    results = solve_steady(params)

    print_summary(results, params)
    plot_final_solution(results, params)


if __name__ == "__main__":
    main()
