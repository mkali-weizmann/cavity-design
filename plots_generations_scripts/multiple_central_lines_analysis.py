# This file


import time

import matplotlib.pyplot as plt
import numpy as np

from cavity import *

lambda_0_laser = 1064e-9
from matplotlib import use
from matplotlib.lines import Line2D
# use('TkAgg')
# %%

# params = np.array([[-5.0000000000e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j, -0.0000000000e+00-1.j,  5.0000281233e-03+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j],
#                    [ 6.5057257992e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  7.9679319133e-03+0.j,  7.9679319133e-03+0.j,  1.0000000000e+00+0.j,  3.0114515984e-03+0.j,  1.7600000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,  1.7600000000e+00+0.j,  5.5000000000e-06+0.j,  1.0000000000e-06+0.j,  4.6060000000e+01+0.j,  1.1700000000e-05+0.j,  3.0000000000e-01+0.j,  1.0000000000e-02+0.j,  1.0000000000e-04+0.j,  9.9989900000e-01+0.j,   np.nan+0.j,  1.0000000000e+00+0.j],
#                    [ 3.0801145160e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.5039504639e-01+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j]])
# # # swap first and third rows of params:
# params[[0, 2]] = params[[2, 0]]
# params[1, [4, 5]] = params[1, [5, 4]]
# params[1, 3] += 1j


params = [OpticalElementParams(surface_type=SurfacesTypes.curved_mirror, x=3.080114516e-01      , y=0                    , z=0                    , t=0                    , p=0                    , r_1=1.5039504639e-01     , r_2=np.nan               , curvature_sign=1.0, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=np.nan               , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=np.nan               , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=np.nan               , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )),
          OpticalElementParams(surface_type=SurfacesTypes.thick_lens,    x=6.5057257992e-03     , y=0                    , z=0                    , t=0                    , p=1e+00 * np.pi        , r_1=7.967931913299999e-03, r_2=7.967931913299999e-03, curvature_sign=1.0, T_c=3.0114515984e-03     , n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=1.76e+00             , alpha_expansion=5.5e-06              , beta_surface_absorption=1e-06                , kappa_conductivity=4.606e+01            , dn_dT=1.17e-05             , nu_poisson_ratio=3e-01                , alpha_volume_absorption=1e-02                , intensity_reflectivity=1e-04                , intensity_transmittance=9.99899e-01          , temperature=np.nan               )),
          OpticalElementParams(surface_type=SurfacesTypes.curved_mirror, x=-5e-03               , y=0                    , z=0                    , t=0                    , p=-1e+00 * np.pi       , r_1=5.0000281233e-03     , r_2=np.nan               , curvature_sign=1.0, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=np.nan               , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=np.nan               , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=np.nan               , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]
params
# %%



cavity = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=lambda_0_laser,
                            names=['Right mirror', 'Lens', 'Left Mirror'],
                            set_central_line=True,
                            set_mode_parameters=True,
                            set_initial_surface=False,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4, use_brute_force_for_central_line=True)
cavity.plot()
# plt.show()
perturbed_cavity = perturb_cavity(cavity, (0, 3), 1e-6)

# theta_initial_guess, phi_initial_guess = perturbed_cavity.default_initial_angles
# initial_parameters = np.array([0, theta_initial_guess, 0, phi_initial_guess])
# def f_angle(d_angle, initial_parameters):
#     diff = perturbed_cavity.f_roots(np.array([initial_parameters[0], initial_parameters[1], initial_parameters[2], initial_parameters[3] + d_angle[0]]))
#     angles_diff = diff[3]
#     return angles_diff
#
# def f_position(d_position, initial_parameters):
#     diff = perturbed_cavity.f_roots(np.array(
#         [initial_parameters[0], initial_parameters[1], initial_parameters[2] + d_position[0], initial_parameters[3]]))
#     positions_diff = diff[2]
#     return positions_diff
#
#
# def bisection_root(f, a, b, epsx, epsf):
#     sign_f_b = np.sign(f(b))
#     h = (b-a) / 2
#     x_mid = a + h #(=0.5*(a+b))
#     f_mid = f(x_mid)
#     sign_f_mid = np.sign(f_mid)
#     if h < epsx or abs(f_mid) < epsf: # Break recursion if desired approximation achieved:
#         return x_mid
#     elif sign_f_mid * sign_f_b == 1:
#         return bisection_root(f, a , x_mid, epsx, epsf)
#     elif sign_f_mid * sign_f_b == -1:
#         return bisection_root(f, x_mid, b, epsx, epsf)
#     else:
#         raise Exception('Debug me')
# # %%
# print("With initial guess: ", perturbed_cavity.f_roots(initial_parameters))
# for i in range(3500):
#     # Extreme brute force
#     if i % 500 == 0:
#         print(i)
#     corrected_angle = optimize.fsolve(f_angle, x0=0, args=(initial_parameters,), xtol=1e-10)  # bisection_root(lambda x: f_angle(x, initial_parameters), -d_angle, d_angle, 1e-12, 1e-12)
#     initial_parameters[3] += corrected_angle
#     corrected_position = optimize.fsolve(f_position, x0=0, args=(initial_parameters,))  # bisection_root(lambda x: f_position(x, initial_parameters), -d_position, d_position, 1e-12, 1e-12)
#     initial_parameters[2] += corrected_position
#     print("initial_parameters: ", initial_parameters)
#     print(f"after position correction: {perturbed_cavity.f_roots(initial_parameters)}")
#
#
initial_parameters = np.array([0,  0, 9.883221063772772e-05, -3.142248804294062])
# %%
# k_vector = unit_vector_of_angles(0, initial_parameters[3])
# origin = perturbed_cavity.surfaces[0].origin
# rays = Ray(origin=origin, k_vector=k_vector)
# central_line_brute_force = perturbed_cavity.trace_ray(rays)
# central_line_smart = perturbed_cavity.trace_ray(perturbed_cavity.central_line[0])
# # central_line_brute_force[-1].length = central_line_brute_force[0].length
# central_line_smart[-1].length = central_line_smart[0].length
# y_lim = 5e-4
# blue_ray_index = 2
#
# fig, ax = plt.subplots(figsize=(15, 10)) #  2, 2,
# perturbed_cavity.plot(ax=ax, plot_central_line=False)
# # central_line_brute_force[0].plot(ax=ax[0], color='green', label='initial')
# for i in range(len(central_line_brute_force)-1):
#     central_line_brute_force[i].plot(ax=ax, color='k')
#     central_line_smart[i].plot(ax=ax, color='r')
# central_line_brute_force[-1].plot(ax=ax, color='k', linestyle='--')
# central_line_smart[-1].plot(ax=ax, color='r', linestyle='--')
# # central_line_brute_force[blue_ray_index].plot(ax=ax[0], color='blue', linestyle='--', label='final')
# custom_lines = [Line2D([0], [0], color='k', lw=1, linestyle='-'),
#                 Line2D([0], [0], color='r', lw=1, linestyle='-'),
#                 Line2D([0], [0], color='k', lw=1, linestyle='--'),
#                 Line2D([0], [0], color='r', lw=1, linestyle='--')]
# ax.legend(custom_lines, ['Brute Force', 'Smart', 'Brute Force next roundtrip', 'Smart next roundtrip'], loc='upper right')
# ax.scatter(perturbed_cavity.surfaces[0].origin[0], perturbed_cavity.surfaces[0].origin[1])
# ax.scatter(perturbed_cavity.surfaces[3].origin[0], perturbed_cavity.surfaces[3].origin[1])
# ax.set_ylim(-y_lim, y_lim)
#
#
# # perturbed_cavity.plot(ax=ax[0, 1], plot_central_line=False)
# # # central_line_brute_force[0].plot(ax=ax[1], color='green', label='initial')
# # for i in range(len(central_line_brute_force)):
# #     central_line_brute_force[i].plot(ax=ax[0, 1], color='k')
# #     central_line_smart[i].plot(ax=ax[0, 1], color='r')
# # # central_line_brute_force[blue_ray_index].plot(ax=ax[1], color='blue', linestyle='--', label='final')
# # ax[0, 1].scatter(perturbed_cavity.surfaces[0].origin[0], perturbed_cavity.surfaces[0].origin[1])
# # ax[0, 1].scatter(perturbed_cavity.surfaces[3].origin[0], perturbed_cavity.surfaces[3].origin[1])
# # x_center = perturbed_cavity.surfaces[3].origin[0]  # central_line_smart[1].origin[0]
# # y_center = perturbed_cavity.surfaces[3].origin[1]  # central_line_smart[1].origin[1]
# # ax[0, 1].set_xlim(x_center-1e-14, x_center+1e-14)
# # ax[0, 1].set_ylim(y_center-1e-14, y_center+1e-14)
# # ax[0, 1].set_title("Zoom on the origin (sphere's center) of the left mirror ")
#
# # perturbed_cavity.plot(ax=ax[1, 0], plot_central_line=False)
# # # central_line_brute_force[0].plot(ax=ax[1], color='green', label='initial')
# # for i in range(len(central_line_brute_force)-1):
# #     central_line_brute_force[i].plot(ax=ax[1, 0], color='k')
# #     central_line_smart[i].plot(ax=ax[1, 0], color='r')
# # central_line_brute_force[-1].plot(ax=ax[1, 0], color='k', linestyle='--')
# # central_line_smart[-1].plot(ax=ax[1, 0], color='r', linestyle='--')
# #
# # # central_line_brute_force[blue_ray_index].plot(ax=ax[1], color='blue', linestyle='--', label='final')
# # ax[1, 0].scatter(perturbed_cavity.surfaces[0].origin[0], perturbed_cavity.surfaces[0].origin[1])
# # ax[1, 0].scatter(perturbed_cavity.surfaces[3].origin[0], perturbed_cavity.surfaces[3].origin[1])
# # x_center = central_line_brute_force[1].origin[0]
# # y_center = central_line_brute_force[1].origin[1]
# # ax[1, 0].set_xlim(x_center-4e-9, x_center+4e-9)
# # ax[1, 0].set_ylim(y_center-3e-12, y_center+3e-12)
# # ax[1, 0].set_title('Zoom on the brute_force central\nline in two different roundtrips')
# #
# # perturbed_cavity.plot(ax=ax[1, 1], plot_central_line=False)
# # # central_line_brute_force[0].plot(ax=ax[1], color='green', label='initial')
# # for i in range(len(central_line_brute_force)-1):
# #     central_line_brute_force[i].plot(ax=ax[1, 1], color='k')
# #     central_line_smart[i].plot(ax=ax[1, 1], color='r')
# # central_line_brute_force[-1].plot(ax=ax[1, 1], color='k', linestyle='--')
# # central_line_smart[-1].plot(ax=ax[1, 1], color='r', linestyle='--')
# # ax[1, 1].scatter(perturbed_cavity.surfaces[0].origin[0], perturbed_cavity.surfaces[0].origin[1])
# # ax[1, 1].scatter(perturbed_cavity.surfaces[3].origin[0], perturbed_cavity.surfaces[3].origin[1])
# # x_center = central_line_smart[1].origin[0]
# # y_center = central_line_smart[1].origin[1]
# # ax[1, 1].set_xlim(x_center-1e-12, x_center+1e-12)
# # ax[1, 1].set_ylim(y_center-3e-12, y_center+3e-12)
# # ax[1, 1].set_title('Zoom on the smart central_line\nin two different roundtrips')
# #
# # fig.tight_layout()
# plt.savefig('figures/two possible solutions for central line.svg', dpi=300, bbox_inches='tight')
# plt.show()
# %%
# Show that there are multiple solutions (the 0 crossings):
# color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# fig, ax = plt.subplots(figsize=(15, 10))
# for j, shift_value in enumerate(np.logspace(-8, -5, 3)):
#     perturbed_cavity = perturb_cavity(cavity, (0, 3), shift_value, set_mode_parameters=False)
#     default_phi = perturbed_cavity.default_initial_angles[1]
#     phis = np.linspace(-1.1e-3, 1.1e-3, 1000)
#     distances = np.zeros_like(phis)
#     for i, phi in enumerate(phis):
#         d = perturbed_cavity.f_roots_standing_wave(np.array([0, np.pi+phi]))
#         distances[i] = d[1]
#     plt.plot(phis, distances, label=f"big mirrors tilt: {shift_value:.1e} [rad]", color=color_cycle[j])
#     plt.axvline(x=default_phi - np.pi, color=color_cycle[j])
# plt.axhline(0, label='0 crossing', color='k')
# plt.xlabel('central line tilt [rad]')
# plt.ylabel("distance to the sphere's center of the small mirror [m]")
# plt.grid()
# plt.legend()
# plt.title('distance between central line and the origin of the end sphere\nfor central line that starts at the origin'
#           'of the first sphere - for multiple tilts')
# plt.savefig('figures/astigmatism_0_crossings.svg', dpi=300, bbox_inches='tight')
# plt.show()

# %% Compare overlap of different modes:
# tilt = 0e-6
# perturbed_cavity_1 = perturb_cavity(cavity, (0, 3), tilt)
# perturbed_cavity_2 = perturb_cavity(cavity, (0, 3), tilt)
#
# def f_reduced(phi):
#     z, y = perturbed_cavity_1.f_roots_standing_wave(np.array([0, phi + np.pi]))
#     if np.isnan(y):
#         y = np.inf * np.sign(phi)
#     return y
#
# default_phi = perturbed_cavity_1.default_initial_angles[1]
# phis = np.linspace(-1e-3, 1e-3, 1000)
# distances = np.zeros_like(phis)
# for i, phi in enumerate(phis):
#     d = f_reduced(phi)  # perturbed_cavity_1.f_roots_standing_wave(np.array([0, np.pi+phi]))
#     distances[i] = d  # [1]
#
# if True:
#     fig, ax = plt.subplots(figsize=(15, 10))
#     plt.plot(phis, distances)
#     plt.grid()
#     plt.show()
#
#
# solution_phi_1 = optimize.brentq(f_reduced, 5e-4, 7.5e-4)  # 2.5e-4, 5e-4
# solution_phi_2 = optimize.brentq(f_reduced, -7.5e-4, -5e-4)  # 3.5e-4, 8e-4
#
# origin = perturbed_cavity_1.surfaces[0].origin
#
# cavities = [perturbed_cavity_1, perturbed_cavity_2]
# solutions = [solution_phi_1, solution_phi_2]
#
#
# for i in range(2):
#     k_vector = unit_vector_of_angles(0, solutions[i])
#
#     ray_for_intersection = Ray(origin=origin, k_vector=-k_vector)
#
#     ray_origin_on_first_surface = cavity.physical_surfaces[0].find_intersection_with_ray(ray_for_intersection)
#
#     solution_ray = Ray(origin=ray_origin_on_first_surface, k_vector=k_vector)
#
#     ray_history = perturbed_cavity_1.trace_ray(solution_ray)
#
#     for j, arm in enumerate(cavities[i].arms):
#         arm.central_line = ray_history[j]
#     cavities[i].central_line_successfully_traced = True
#     cavities[i].set_mode_parameters()
#
# overlap = calculate_cavities_overlap(perturbed_cavity_1, perturbed_cavity_2)
# print(f"Overlap: {np.abs(overlap):.2f}")
#
# fig, ax = plt.subplots(figsize=(15, 10))
# plt.plot(phis, distances)
# plt.axvline(x=solution_phi_1, color='r', label='solution 1')
# plt.axvline(x=solution_phi_2, color='g', label='solution 2')
# plt.axvline(x=default_phi - np.pi, color='k', label='default')
# plt.xlabel('central line tilt [rad]')
# plt.grid()
# plt.title(f"overlap: {np.abs(overlap):.2f} between the two modes")
# plt.savefig('figures/astigmatism_overlap.svg', dpi=300, bbox_inches='tight')
# plt.show()

# %% Repeating the same for an ideal thick lens:

params = [OpticalElementParams(surface_type=SurfacesTypes.curved_mirror,x=3.080114516e-01      , y=0                    , z=0                    , t=0                    , p=0                    , r_1=1.5039504639e-01     , r_2=np.nan               , curvature_sign=1.0, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=np.nan               , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=np.nan               , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=np.nan               , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )),
          OpticalElementParams(surface_type=SurfacesTypes.thick_lens,   x=6.5057257992e-03     , y=0                    , z=0                    , t=0                    , p=1e+00 * np.pi        , r_1=7.967931913299999e-03, r_2=7.967931913299999e-03, curvature_sign=1.0, T_c=3.0114515984e-03     , n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=1.76e+00             , alpha_expansion=5.5e-06              , beta_surface_absorption=1e-06                , kappa_conductivity=4.606e+01            , dn_dT=1.17e-05             , nu_poisson_ratio=3e-01                , alpha_volume_absorption=1e-02                , intensity_reflectivity=1e-04                , intensity_transmittance=9.99899e-01          , temperature=np.nan               )),
          OpticalElementParams(surface_type=SurfacesTypes.curved_mirror,x=-5e-03               , y=0                    , z=0                    , t=0                    , p=-1e+00 * np.pi       , r_1=5.0000281233e-03     , r_2=np.nan               , curvature_sign=1.0, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=np.nan               , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=np.nan               , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=np.nan               , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]
params

cavity = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=lambda_0_laser,
                            names=['Right mirror', 'Lens', 'Left Mirror'],
                            set_central_line=True,
                            set_mode_parameters=True,
                            set_initial_surface=False,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4,
                            use_brute_force_for_central_line=True,
                            use_paraxial_ray_tracing=False)

# Show that there are multiple solutions (the 0 crossings):
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(figsize=(15, 10))
for j, shift_value in enumerate(np.logspace(-8, -5, 3)):
    perturbed_cavity = perturb_cavity(cavity, (0, 3), shift_value, set_mode_parameters=False)
    default_phi = perturbed_cavity.default_initial_angles[1]
    phis = np.linspace(-2.1e-3, 2.1e-3, 8)
    distances = np.zeros_like(phis)
    for i, phi in enumerate(phis):
        d = perturbed_cavity.f_roots_standing_wave(np.array([0, np.pi+phi]))
        distances[i] = d[1]
    plt.plot(phis, distances, label=f"big mirrors tilt: {shift_value:.1e} [rad]", color=color_cycle[j])
    plt.axvline(x=default_phi - np.pi, color=color_cycle[j])
plt.axhline(0, label='0 crossing', color='k')
plt.xlabel('central line tilt [rad]')
plt.ylabel("distance to the sphere's center of the small mirror [m]")
plt.grid()
plt.legend()
plt.title('distance between central line and the origin of the end sphere\nfor central line that starts at the origin'
          'of the first sphere - for multiple tilts')
# plt.savefig('figures/astigmatism_0_crossings.svg', dpi=300, bbox_inches='tight')
plt.show()
