# %%  # Generate an aspheric lens, where the rays hit the flat surface first:
from matplotlib import use
use('TkAgg')  # or 'Qt5Agg', 'Agg', etc. depending on your environment
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


@dataclass(frozen=True)
class LensParams:
    n: float      # refractive index (assumed > 1)
    f: float      # distance outside the lens
    Tc: float     # center thickness parameter in your equation
    eps: float = 1e-12  # numerical guard


def constraint_g(theta: float, y: float, x: float, p: LensParams) -> float:
    """
    Constraint equation:
        f * (n sinθ)/sqrt(1 - n^2 sin^2θ) + (Tc + x) tanθ - y = 0
    """
    s = np.sin(theta)
    c = np.cos(theta)

    # Domain: 1 - n^2 sin^2θ > 0 (no TIR at the flat face in this model)
    rad = 1.0 - (p.n * s) ** 2
    if rad <= 0.0:
        # Return a large value with consistent sign-ish behavior; root finder will avoid.
        return np.sign(rad) * 1e6

    term1 = p.f * (p.n * s) / np.sqrt(rad)
    term2 = (p.Tc - x) * (s / c)  # tanθ
    return term1 + term2 - y


def find_theta(y: float, x: float, p: LensParams, theta_hint: float = 0.0) -> float:
    """
    Solve constraint_g(theta; y, x) = 0 for theta.

    We restrict theta to (-theta_max, theta_max) where theta_max = arcsin(1/n) - margin,
    to keep sqrt(1 - n^2 sin^2θ) real.

    Uses bracketing + root_scalar(brentq) with an expanding bracket around theta_hint.
    """
    if abs(y) < p.eps:
        return 0.0

    # Maximum physically allowed |theta| in this model (avoids sqrt singularity)
    theta_max = np.arcsin(min(1.0, 1.0 / p.n)) - 1e-9
    theta_max = max(theta_max, 1e-6)  # guard in case n ~ 1

    # Clamp hint into domain
    theta_hint = float(np.clip(theta_hint, -theta_max, theta_max))

    # If y>0 we typically want theta>0; if y<0, theta<0 (helps bracketing)
    preferred_sign = 1.0 if y >= 0 else -1.0
    if theta_hint == 0.0:
        theta_hint = preferred_sign * min(0.1, 0.5 * theta_max)

    def g(th):
        return constraint_g(th, y, x, p)

    # Build an expanding bracket around theta_hint until sign change
    delta = min(0.05, 0.25 * theta_max)
    a = np.clip(theta_hint - delta, -theta_max, theta_max)
    b = np.clip(theta_hint + delta, -theta_max, theta_max)

    ga = g(a)
    gb = g(b)

    # Expand bracket
    for _ in range(60):
        if np.isfinite(ga) and np.isfinite(gb) and ga * gb <= 0.0:
            break
        delta *= 1.5
        a = np.clip(theta_hint - delta, -theta_max, theta_max)
        b = np.clip(theta_hint + delta, -theta_max, theta_max)
        ga = g(a)
        gb = g(b)
    else:
        # As a fallback, try a global bracket over the full domain
        a, b = -theta_max, theta_max
        ga, gb = g(a), g(b)
        if not (np.isfinite(ga) and np.isfinite(gb) and ga * gb <= 0.0):
            raise RuntimeError(
                "Could not bracket a root for theta. Likely no real solution for this (y, x) "
                "under the constraint (e.g., beyond model domain / TIR / geometry mismatch)."
            )

    sol = root_scalar(g, bracket=(a, b), method="brentq", xtol=1e-12, rtol=1e-12, maxiter=200)
    if not sol.converged:
        raise RuntimeError("Theta solve did not converge.")
    return float(sol.root)


def rhs_factory(p: LensParams):
    """
    Returns an RHS function for solve_ivp that keeps a running theta hint for speed/stability.
    """
    state = {"theta_hint": 0.0}

    def rhs(y: float, x_arr: np.ndarray) -> np.ndarray:
        x = float(x_arr[0])

        # User assumption: Tc + x > 0 (we enforce numerically)
        if p.Tc + x <= p.eps:
            raise RuntimeError("Encountered Tc + x <= 0; violates the stated assumption Tc + x > 0.")

        theta = find_theta(y, x, p, theta_hint=state["theta_hint"])
        state["theta_hint"] = theta

        s = np.sin(theta)
        c = np.cos(theta)

        denom = p.n * c - 1.0
        if abs(denom) < p.eps:
            raise RuntimeError("Encountered n*cos(theta) - 1 ~ 0, slope diverges (model singularity).")

        dxdy = (p.n * s) / denom
        return np.array([dxdy])

    return rhs


def solve_profile(
    p: LensParams,
    y_max: float,
    *,
    n_points: int = 2000,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-11,
) -> Tuple[np.ndarray, np.ndarray, solve_ivp]:
    """
    Solve x(y) for y in [0, y_max] with x(0)=0 using the coupled implicit theta constraint.
    """
    rhs = rhs_factory(p)

    sol = solve_ivp(
        fun=rhs,
        t_span=(0.0, float(y_max)),
        y0=np.array([0.0]),
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    y = np.linspace(0.0, y_max, n_points)
    x = sol.sol(y)[0]
    return y, x, sol



if __name__ == "__main__":
    # Example (edit to your values)
    params = LensParams(n=1.5, f=20.0, Tc=5.0)  #(n=1.5168, f=45.23, Tc=7.24)#

    y, x, sol = solve_profile(params, y_max=12.5, n_points=1500)

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        plt.plot(y, x)
        plt.xlabel("y")
        plt.ylabel("x(y)")
        plt.title("Lens surface from implicit-theta ODE system")
        plt.grid(True)
        plt.show()
    except Exception:
        pass
# %% validate results:
if __name__ == "__main__":
    dx_dy = np.gradient(x, y)
    theta_vals = np.array([find_theta(yy, xx, params) for yy, xx in zip(y, x)])
    s_vals = np.sin(theta_vals)
    c_vals = np.cos(theta_vals)
    term1 = params.f * (params.n * s_vals) / np.sqrt(1.0 - (params.n * s_vals) ** 2)
    term2 = (params.Tc + x) * (s_vals / c_vals)
    g_vals = term1 + term2 - y
    # first is the extracted derivative from the results and second column
    # is the derivative's value according to the differential equation so they should
    # be equal. Third column is the relation between x, y, and theta and should be ~0 when the condition is met
    stacked = np.vstack([dx_dy, (params.n * s_vals) / (params.n * c_vals - 1.0), g_vals]).T

    # Specifically for n=1.511, f=0.04523, Tc=0.00724, edmunds optics element:
    k, C, E, F = -8.00424e-1, 3.869969e-2, 1.643994e-6, 5.887865e-10
    x_edmund = C * y**2 / (1 + np.sqrt(1 - (1 + k) * C**2 * y**2)) + E * y**4 + F * y**6

# %% Fit a polynomial to x as a function of y^2, with only even powers:
    from numpy.polynomial import Polynomial
    y_squared = y**2
    degree = 8  # even degree
    # Fit only even powers: we fit a polynomial in y^2
    p_fit = Polynomial.fit(y_squared, x, deg=degree//2)
    x_fit = p_fit(y_squared)

    plt.plot(x, y, label='Original Data')
    plt.plot(x_fit, y, label='Fitted Polynomial', linestyle='--')

    plt.plot(x_edmund, y, label='Edmunds face', linestyle='--')
    # plt.plot(x_fit, -y, linestyle='--')  # Mirror for negative y
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("Aspheric Surface Fit with Even Polynomial")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"polynomial coefficients (even powers only): {p_fit.convert().coef}")
    plt.close()
    residual = x_edmund - x
    residual_fit = Polynomial.fit(y_squared, residual, deg=degree//2)
    plt.plot(y, x_edmund - x)
    plt.plot(y, residual_fit(y_squared), linestyle='--')
    print(f"Residual polynomial coefficients (even powers only):  = {residual_fit.convert().coef}")
    plt.show()





# #  %%  Generate an aspheric lens, where the rays hit the curved surface first
# @dataclass(frozen=True)
# class AsphereParams:
#     n: float      # refractive index (n > 1)
#     f: float      # focal length
#
#
# def rhs(y: float, x_arr: np.ndarray, p: AsphereParams) -> np.ndarray:
#     """
#     ODE:
#     dx/dy = ( (y/(f+x)) / ( n*sqrt(1 + (y/(f+x))^2) - 1 ) )
#     """
#     x = float(x_arr[0])
#     a = y / (p.f + x)
#     denominator = p.n * np.sqrt(1.0 + a*a) - 1.0
#
#     return np.array([a / denominator])
#
#
# def solve_asphere(
#     p: AsphereParams,
#     y_max: float,
#     *,
#     n_points: int = 2000,
#     method: str = "RK45",
#     rtol: float = 1e-9,
#     atol: float = 1e-12,
#     max_step: Optional[float] = None,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Solve x(y) for the aspheric surface.
#     """
#     sol = solve_ivp(
#         fun=lambda y, x: rhs(y, x, p),
#         t_span=(0.0, y_max),
#         y0=np.array([0.0]),
#         method=method,
#         rtol=rtol,
#         atol=atol,
#         dense_output=True,
#     )
#
#     if not sol.success:
#         raise RuntimeError(sol.message)
#
#     y = np.linspace(0.0, y_max, n_points)
#     x = sol.sol(y)[0]
#     return y, x
#
#
#
# params = AsphereParams(
#     n=1.7,
#     f=20.0,
# )
#
# y, x = solve_asphere(params, y_max=12.5)
#
# # Optional plot
# try:
#     import matplotlib.pyplot as plt
#     plt.plot(x, y)
#     plt.plot(x, -y)
#     plt.xlabel("x")
#     plt.ylabel("y(x)")
#     plt.title("Aspheric surface from exact Snell-law ODE")
#     # Set aspect ratio to 1:1
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.grid(True)
#     plt.show()
# except Exception:
#     pass
#
# # %% validate_solution
# dx_dy = np.gradient(x, y)
# a = y / (params.f + x)
# denominator = params.n * np.sqrt(1.0 + a*a) - 1
# rhs_values = a / denominator
# # first is the extracted derivative from the results and second column
# # is the derivative's value according to the differential equation so they should
# # be equal.
# stacked = np.vstack([dx_dy, rhs_values]).T