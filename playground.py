from matplotlib import use
use('TkAgg')
from simple_analysis_scripts.potential_analysis.analyze_potential import *
from delete_me import solve_2d_direct_ground_state

cavity_disordered = fabry_perot_generator(radii=(1, 1), NA=0.1, lambda_0_laser=LAMBDA_0_LASER)
cavity = Cavity(surfaces=[cavity_disordered.surfaces[1], cavity_disordered.surfaces[0]],
                p_is_trivial=True,
                t_is_trivial=True,
                use_paraxial_ray_tracing=False,
                set_mode_parameters=True,
                lambda_0_laser=LAMBDA_0_LASER)
results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=100, phi_max=0.3, print_tests=False)
# fig, ax = plot_results(results_dict)
# plt.show()
hessian_value = hessian(cavity=cavity, n_rays=1, phi_max=0)[0, 0]

# DRAFT
r, E0, psi0, H = solve_cavity_eigenstate(cavity=cavity, phi_max=0.1)
# solve_2d_direct_ground_state()
plt.figure(figsize=(8, 5))
plt.plot(r, psi0, label="Numerical $\\psi_0(r)$ chatGPT")
# Plot to verify the origin is well-behaved
plt.plot(r, psi_0, label=r'Numerical $\psi_0(r)$ gemini')
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

