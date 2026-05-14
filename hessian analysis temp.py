from matplotlib import use
use('TkAgg')
from cavity_design import *
# from delete_me import solve_2d_direct_ground_state

# cavity_disordered = fabry_perot_generator(radii=(1, 1), NA=0.1, lambda_0_laser=LAMBDA_0_LASER)


u = 2.8e-3
R_0 = 1
R_1 = 1

cavity_disordered = fabry_perot_generator((R_0, R_1), unconcentricity=u, lambda_0_laser=LAMBDA_0_LASER, use_paraxial_ray_tracing=False)
cavity = Cavity(surfaces=[cavity_disordered.surfaces[1], cavity_disordered.surfaces[0]],
                p_is_trivial=True,
                t_is_trivial=True,
                use_paraxial_ray_tracing=False,
                set_mode_parameters=True,
                lambda_0_laser=LAMBDA_0_LASER)
hessian_ray_tracing_value = hessian_ray_tracing(cavity=cavity, n_rays=1, phi_max=0.1)[0, 0]
hessian_ABCD_matrices_value = hessian_ABCD_matrices(cavity=cavity, n_rays=1, phi_max=0.1)[0, 0]
hessian_analytical = -R_1 / ((R_0 + R_1) * R_0)

dp = 1e-3
slightly_shifted_ray = Ray(origin=cavity.surfaces[0].parameterization(0, dp),
                           k_vector=-cavity.surfaces[0].normal_at_a_point(cavity.surfaces[0].parameterization(0, dp)))
slightly_shifted_ray_propagated = cavity.propagate_ray(ray=slightly_shifted_ray, n_arms = len(cavity.arms) // 2)
landing_point = slightly_shifted_ray_propagated[-1].origin
landing_point_parameterization = cavity.surfaces[-1].get_parameterization(landing_point)[1]
jacobian = dp / landing_point_parameterization

results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=10, phi_max=0.01, print_tests=False)
a_2_numerical = results_dict['polynomial_residuals_mirror'].coef[1]
a_2_analytical = u / (2 * R_1 ** 2)

hessian_normalized = hessian_ABCD_matrices_value * jacobian ** 2

w_squared_ABCD = cavity.arms[0].mode_parameters_on_surface_1.spot_size[0] ** 2
w_squared_geometrical = cavity.lambda_0_laser / (np.pi * np.sqrt(2 * np.abs(hessian_ABCD_matrices_value) * a_2_analytical))
w_squared_geometrical_normalized = cavity.lambda_0_laser / (np.pi * np.sqrt(2 * np.abs(hessian_normalized) * a_2_analytical))
# w_squared_analytical_optics = R * cavity.lambda_0_laser / np.pi * np.sqrt(2 * R / u)

energy_level_hessian_only = cavity.lambda_0_laser ** 2 / (2 * np.pi ** 2 * w_squared_ABCD * hessian_normalized)
energy_level_hessian_and_potential = np.sqrt(a_2_analytical / (-2 * hessian_normalized)) * cavity.lambda_0_laser / np.pi
energy_level_spot_size_and_potential = a_2_analytical * w_squared_ABCD

# plot_results(results_dict)
# plt.show()
print(f'Potential quadratic coefficient: {a_2_numerical:.3e} m^-1')
print(f'Analytical potential quadratic coefficient: {a_2_analytical:.3e} m^-1')
print(f'Hessian ray tracing: {hessian_ray_tracing_value}')
print(f'Hessian ABCD matrices: {hessian_ABCD_matrices_value}')
print(f'Analytical Hessian: {hessian_analytical}')
print(f'Numerical spot size squared: {w_squared_ABCD:.3e} m^2')
print(f'Analytical spot size potential squared: {w_squared_geometrical:.3e} m^2')
print(f'Analytical spot size potential squared normalized: {w_squared_geometrical_normalized:.3e} m^2')
print(f'Spot sizes squared ratio: {w_squared_ABCD / w_squared_geometrical:.5f}')
print(f'Energy level from Hessian only: {energy_level_hessian_only:.3e} m')
print(f'Energy level from Hessian and potential: {energy_level_hessian_and_potential:.3e} m')
print(f'Energy level from spot size and potential: {energy_level_spot_size_and_potential:.3e} m')

energy_level_value = energy_level(cavity=cavity, hessian_method ='ABCD_matrices')
print(energy_level_value)
results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=100, phi_max=0.3, print_tests=False)
# fig, ax = plot_results(results_dict)
# plt.show()
hessian_value = hessian(cavity=cavity, n_rays=1, phi_max=0)[0, 0]

# DRAFT
r, E_0, psi_0, H = solve_cavity_eigenstate(cavity=cavity, phi_max=0.005)
# solve_2d_direct_ground_state()
plt.figure(figsize=(8, 5))
plt.plot(r, psi_0, label="Numerical $\\psi_0(r)$ chatGPT")
plt.title("2D Harmonic Oscillator Ground State (Direct Method)")
plt.xlabel("r")
plt.ylabel(r"$\psi_0(r)$")
plt.grid(True)
plt.legend()
plt.show()
plt.xlabel("r")
plt.ylabel("$\\psi_0(r)$")
plt.title("2D Harmonic Oscillator Ground State")
plt.legend()
plt.grid(True)
plt.show()

# %%
from cavity_design.analyze_potential import *

import numpy as np
import matplotlib.pyplot as plt


# Parameters
m = 3.0
omega = 2.0
r_max = 3 * np.sqrt(H_BAR / m * omega)  # Choose r_max based on the classical turning point
n = 1000

# Harmonic potential
def V_ho(r):
    return 0.5 * m * omega**2 * r**2

# Numerical solution
r, E0_num, psi0_num, H = ground_state_2d_radial_polar(r_max, V_ho, m, n)

# Exact solution
E0_exact = omega
psi0_exact = np.sqrt(2 * m * omega / H_BAR) * np.exp(-0.5 * m * omega * r**2 / H_BAR)

# Align overall sign
if np.dot(psi0_num, psi0_exact) < 0:
    psi0_num = -psi0_num

# Energy comparison
print(f"Numerical E0 = {E0_num:.10f}")
print(f"Exact     E0 = {E0_exact:.10f}")
print(f"Absolute error = {abs(E0_num - E0_exact):.3e}")

# Normalization check
norm = np.trapezoid(np.abs(psi0_num)**2 * r, r)
print(f"Radial normalization = {norm:.10f}")

# Wavefunction error
wavefunc_error = np.sqrt(np.trapezoid((psi0_num - psi0_exact)**2 * r, r))
print(f"Wavefunction error = {wavefunc_error:.3e}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(r, psi0_num, label=r"Numerical $\psi_0(r)$")
plt.plot(r, psi0_exact, "--", label=r"Exact $\psi_0(r)$")
plt.xlabel("r")
plt.ylabel(r"$\psi_0(r)$")
plt.title("2D Harmonic Oscillator Ground State")
plt.legend()
plt.grid(True)
plt.show()