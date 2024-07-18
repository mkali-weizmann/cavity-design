from cavity import *

x_coarse = 0.0000000000e+00
x = 0.0000000000e+00
x_fine = 1.3234889801e-23
t_coarse = 0.0000000000e+00
t = 0.0000000000e+00
t_fine = 1.3234889801e-23
y_coarse = 0.0000000000e+00
y = 0.0000000000e+00
y_fine = 1.3234889801e-23
p_coarse = 0.0000000000e+00
p = 0.0000000000e+00
p_fine = 1.3234889801e-23
use_paraxial_ray_tracing = True
x_lim = -5.0000000000e+00
y_lim_0 = -7.9240000000e+00
y_lim_1 = -5.0000000000e+00
y_lim_2 = -2.0000000000e+00
print_input_parameters = True
true_for_mirror_false_for_origin = False

lambda_0_laser = 1064e-9

# New cavity params:
params = np.array([[-5.0000000000e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j, -0.0000000000e+00-1.j,  5.0000281233e-03+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j],
                   [ 6.5057257992e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  7.9679319133e-03+0.j,  7.9679319133e-03+0.j,  1.0000000000e+00+0.j,  3.0114515984e-03+0.j,  1.7600000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,  1.7600000000e+00+0.j,  5.5000000000e-06+0.j,  1.0000000000e-06+0.j,  4.6060000000e+01+0.j,  1.1700000000e-05+0.j,  3.0000000000e-01+0.j,  1.0000000000e-02+0.j,  1.0000000000e-04+0.j,  9.9989900000e-01+0.j,   np.nan+0.j,  1.0000000000e+00+0.j],
                   [ 3.0801145160e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.5039504639e-01+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j]])

# # swap first and third rows of params:
params[[0, 2]] = params[[2, 0]]
params[1, [4, 5]] = params[1, [5, 4]]
params[1, 3] += 1j

cavity = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=lambda_0_laser,
                            names=['Right mirror', 'Lens', 'Left Mirror'],
                            set_central_line=True,
                            set_mode_parameters=True,
                            set_initial_surface=False,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4, use_brute_force_for_central_line=True,
                            use_paraxial_ray_tracing=use_paraxial_ray_tracing)
# cavity.plot()
# plt.show()

# assert not(center_around_second_center and center_between_two_origins), "can not center both around second center and between two centers"

perturbed_cavity = perturb_cavity(cavity, (0, 3), 0)
phi_initial_guess = np.pi
theta_initial_guess = 0
y_initial_guess = 0

x_total = x_coarse + x + x_fine
t_total = t_coarse + t + t_fine + theta_initial_guess
y_total = y_coarse + y + y_fine + y_initial_guess
p_total = p_coarse + p + p_fine + phi_initial_guess

if true_for_mirror_false_for_origin:

    initial_parameters = np.array([x_total, t_total, y_total, p_total])
    _, ray_history = perturbed_cavity.trace_ray_parametric(initial_parameters)
    # diff = perturbed_cavity.f_roots(initial_parameters)

else:
    k_vector = unit_vector_of_angles(t_total, p_total)
    ray = Ray(perturbed_cavity.physical_surfaces[0].origin, k_vector)
    ray_history = perturbed_cavity.trace_ray(ray)
    # diff = perturbed_cavity.f_roots_standing_wave(angles=(t_total, p_total))

fig, axes = plt.subplots(1, 3, figsize=(21, 5))
for ax in axes:
    perturbed_cavity.plot(ax=ax, plot_central_line=False)

    for i, ray in enumerate(ray_history):
        ray.plot(ax=ax, label=i)
    ax.scatter(perturbed_cavity.surfaces[0].origin[0], perturbed_cavity.surfaces[0].origin[1])
    ax.scatter(perturbed_cavity.surfaces[3].origin[0], perturbed_cavity.surfaces[3].origin[1])
    ax.legend()

center_x = perturbed_cavity.surfaces[3].origin[0]
x_lim = 10 ** x_lim
axes[0].set_xlim(center_x - x_lim, center_x + x_lim)
axes[0].set_ylim(-10 ** y_lim_0, 10 ** y_lim_0)

axes[1].set_xlim(perturbed_cavity.surfaces[3].origin[0] - 0.001, perturbed_cavity.surfaces[1].center[0] + 0.001)
axes[1].set_ylim(-10 ** y_lim_1, 10 ** y_lim_1)

axes[2].set_xlim(perturbed_cavity.surfaces[3].origin[0] - 0.001, perturbed_cavity.surfaces[0].origin[0] + 0.01)
axes[2].set_ylim(-10 ** y_lim_2, 10 ** y_lim_2)

plt.show()


# %%

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(figsize=(15, 10))
default_phi = cavity.default_initial_angles[1]
phis = np.linspace(-1e-3, 1e-3, 5)# np.array([0, 2.8e-4, 5.785e-4])
distances = np.zeros_like(phis)
for i, phi in enumerate(phis):
    d = cavity.f_roots_standing_wave(np.array([0, np.pi+phi]))
    distances[i] = d[1]
plt.plot(phis, distances)
plt.axvline(x=default_phi - np.pi)
plt.axhline(0, label='0 crossing', color='k')
plt.xlabel('central line tilt [rad]')
plt.ylabel("distance to the sphere's center of the small mirror [m]")
plt.grid()
plt.legend()
plt.title('distance between central line and the origin of the end sphere\nfor central line that starts at the origin'
          'of the first sphere - for multiple tilts')
# plt.savefig('figures/astigmatism_0_crossings.svg', dpi=300, bbox_inches='tight')
plt.show()