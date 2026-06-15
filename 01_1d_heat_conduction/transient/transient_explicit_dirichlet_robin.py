"""
1D transient heat conduction with internal heat generation
and a convective boundary condition at x = L.

Numerical method:
    - Cell-centered finite volume method
    - Explicit Forward Euler time integration
    - Dirichlet boundary condition at x = 0
    - Robin / convective boundary condition at x = L
    - Uniform volumetric heat generation
    - Analytical steady-state validation

Physical problem:
    rho cp dT/dt = k d²T/dx² + q_vol

Boundary conditions:
    T(0, t) = T_left
    -k dT/dx(L, t) = h [T(L, t) - T_inf]

Author: David Batet Romero
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ProblemParameters:
    # Geometry
    L: float = 1.5
    area: float = 1.0

    # Material properties
    k: float = 30.0
    rho: float = 1000.0
    cp: float = 10.0

    # Boundary and source terms
    T_left: float = 30.0
    T_inf: float = 30.0
    h: float = 1000.0
    q_vol: float = 1.0e4

    # Time parameters
    dt: float = 0.015
    t_final: float = 1000.0
    T_init: float = 30.0

    # Mesh
    N: int = 150

    # Convergence
    tolerance: float = 1.0e-6


def build_mesh(params: ProblemParameters):
    """
    Build a uniform cell-centered mesh.
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


def compute_coefficients(params: ProblemParameters, dx: float, delta_b: float):
    """
    Compute finite volume conductances and explicit update coefficients.
    """
    k = params.k
    h = params.h
    area = params.area

    # Internal diffusive conductance between neighbouring cell centers
    a_int = k * area / dx

    # Boundary conductance at x = 0
    a_bL = k * area / delta_b

    # Equivalent Robin conductance at x = L
    a_bR = k * h * area / (k + h * delta_b)

    # Control-volume heat capacity
    volume = area * dx
    capacity = params.rho * params.cp * volume

    # Explicit factor multiplying the net heat rate
    explicit_factor = params.dt / capacity

    # Thermal diffusivity and Fourier number
    alpha = params.k / (params.rho * params.cp)
    fourier = alpha * params.dt / dx**2

    # Volumetric source increment
    source_increment = params.q_vol * params.dt / (params.rho * params.cp)

    coefficients = {
        "a_int": a_int,
        "a_bL": a_bL,
        "a_bR": a_bR,
        "explicit_factor": explicit_factor,
        "fourier": fourier,
        "source_increment": source_increment,
    }

    return coefficients


def compute_right_wall_temperature(T_last: float, params: ProblemParameters, delta_b: float):
    """
    Compute the wall temperature at x = L from the Robin boundary condition.
    """
    T_wall = (
        params.k * T_last
        + params.h * delta_b * params.T_inf
    ) / (params.k + params.h * delta_b)

    return T_wall


def explicit_time_step(T: np.ndarray, params: ProblemParameters, coefficients: dict):
    """
    Advance the temperature field by one explicit Forward Euler time step.
    """
    T_new = T.copy()

    a_int = coefficients["a_int"]
    a_bL = coefficients["a_bL"]
    a_bR = coefficients["a_bR"]
    explicit_factor = coefficients["explicit_factor"]
    source_increment = coefficients["source_increment"]
    fourier = coefficients["fourier"]

    # Left boundary-adjacent cell: Dirichlet condition at x = 0
    T_new[0] = (
        T[0]
        + explicit_factor * (
            a_bL * (params.T_left - T[0])
            + a_int * (T[1] - T[0])
        )
        + source_increment
    )

    # Interior cells
    T_new[1:-1] = (
        T[1:-1]
        + fourier * (T[2:] - 2.0 * T[1:-1] + T[:-2])
        + source_increment
    )

    # Right boundary-adjacent cell: Robin condition at x = L
    T_new[-1] = (
        T[-1]
        + explicit_factor * (
            a_int * (T[-2] - T[-1])
            + a_bR * (params.T_inf - T[-1])
        )
        + source_increment
    )

    return T_new


def solve_transient(
    params: ProblemParameters,
    live_plot: bool = True,
    refresh_dt_sim: float = 0.05,
):
    """
    Solve the transient heat conduction problem using explicit Forward Euler.
    """
    x_c, dx, delta_b = build_mesh(params)

    coefficients = compute_coefficients(params, dx, delta_b)

    T = np.full(params.N, params.T_init, dtype=float)
    T_steady = analytical_steady_solution(x_c, params)

    nsteps = int(np.ceil(params.t_final / params.dt))
    save_every = max(int(refresh_dt_sim / params.dt), 1)

    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 4))

        (line_num,) = ax.plot(
            x_c,
            T,
            label="Explicit FVM evolution",
        )

        ax.plot(
            x_c,
            T_steady,
            "--",
            label="Analytical steady solution",
        )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("T [°C]")
        ax.set_title("Temperature evolution — explicit FVM")
        ax.grid(True)
        ax.legend()

        txt = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            va="top",
        )

        fig.tight_layout()

    converged = False
    time = 0.0
    temporal_error = np.inf

    for step in range(nsteps):
        time = (step + 1) * params.dt

        T_new = explicit_time_step(T, params, coefficients)

        temporal_error = np.max(np.abs(T_new - T))

        if live_plot and ((step + 1) % save_every == 0):
            line_num.set_ydata(T_new)
            txt.set_text(
                f"t = {time:.2f} s\n"
                f"ΔT_inf = {temporal_error:.2e} °C"
            )
            fig.canvas.draw_idle()
            plt.pause(0.001)

        T = T_new

        if temporal_error < params.tolerance:
            converged = True
            print(
                f"Steady state reached at t = {time:.3f} s, "
                f"err = {temporal_error:.2e} °C"
            )
            break

    if live_plot:
        line_num.set_ydata(T)
        txt.set_text(
            f"t = {time:.2f} s\n"
            f"ΔT_inf = {temporal_error:.2e} °C"
        )
        fig.canvas.draw()
        plt.ioff()
        plt.show()

    steady_error = np.max(np.abs(T - T_steady))

    results = {
        "x_c": x_c,
        "dx": dx,
        "delta_b": delta_b,
        "T": T,
        "T_steady": T_steady,
        "T_wall_left": params.T_left,
        "T_wall_right": compute_right_wall_temperature(T[-1], params, delta_b),
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

    x_plot = np.r_[0.0, x_c, params.L]
    T_plot = np.r_[results["T_wall_left"], T, results["T_wall_right"]]

    x_exact = np.linspace(0.0, params.L, 500)
    T_exact = analytical_steady_solution(x_exact, params)

    if results["converged"]:
        status = "converged solution"
    else:
        status = f"non-converged solution at t = {results['time']:.1f} s"

    plt.figure(figsize=(7, 4))
    plt.plot(
        x_plot,
        T_plot,
        "o",
        markersize=4,
        markevery=max(params.N // 50, 1),
        label=f"Numerical FVM ({status})",
    )
    plt.plot(
        x_exact,
        T_exact,
        "-",
        label="Analytical steady solution",
    )
    plt.xlabel("x [m]")
    plt.ylabel("T [°C]")
    plt.title("1D explicit heat conduction with internal generation and Robin boundary")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_summary(results: dict, params: ProblemParameters):
    """
    Print a concise numerical summary of the simulation.
    """
    coefficients = results["coefficients"]
    fourier = coefficients["fourier"]

    left_explicit_sum = coefficients["explicit_factor"] * (
        coefficients["a_bL"] + coefficients["a_int"]
    )

    right_explicit_sum = coefficients["explicit_factor"] * (
        coefficients["a_bR"] + coefficients["a_int"]
    )

    print("\n=== Simulation summary ===")
    print(f"Number of control volumes      : {params.N}")
    print(f"Cell size                      : {results['dx']:.6e} m")
    print(f"Time step                      : {params.dt:.6e} s")
    print(f"Final simulated time           : {results['time']:.6f} s")
    print(f"Number of time steps           : {results['nsteps']}")
    print(f"Fourier number                 : {fourier:.6e}")
    print(f"Left explicit coefficient sum  : {left_explicit_sum:.6e}")
    print(f"Right explicit coefficient sum : {right_explicit_sum:.6e}")
    print(f"Converged                      : {results['converged']}")
    print(f"Temporal change L_inf          : {results['temporal_error']:.6e} °C")
    print(f"Steady analytical error L_inf  : {results['steady_error']:.6e} °C")
    print(f"Maximum numerical temperature  : {np.max(results['T']):.6f} °C")
    print(f"Right wall temperature         : {results['T_wall_right']:.6f} °C")

    if fourier > 0.5:
        print(
            "\n[WARNING] Fourier number is larger than the classical "
            "explicit stability limit Fo <= 0.5."
        )


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
        dt=0.015,
        t_final=1000.0,
        T_init=30.0,
        N=150,
        tolerance=1.0e-6,
    )

    results = solve_transient(
        params,
        live_plot=True,
        refresh_dt_sim=0.05,
    )

    print_summary(results, params)
    plot_final_solution(results, params)


if __name__ == "__main__":
    main()
