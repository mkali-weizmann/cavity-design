from cavity import *

lambda_0_laser = 1064e-9
x_coarse = 0.0000000000e+00
x = 0.0000000000e+00
x_fine = 1.3234889801e-23
t_coarse = 0.0000000000e+00
t = 0.0000000000e+00
t_fine = 4.2127962662e-24
y_coarse = 0.0000000000e+00
y = 0.0000000000e+00
y_fine = 1.3234889801e-23
p_coarse = -0.00025
p = 0.0000000000e+00
p_fine = 4.2127962662e-24
use_paraxial_ray_tracing = True
x_lim = -5.0000000000e+00
y_lim = -5.0000000000e+00
center_around_second_center = False
print_input_parameters = True

params = [OpticalElementParams(surface_type=SurfacesTypes.curved_mirror, x=3.080114516e-01      , y=0                    , z=0                    , t=0                    , p=0                    , r_1=1.5039504639e-01     , r_2=np.nan               , curvature_sign=1.0, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=np.nan               , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=np.nan               , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=np.nan               , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )),
          OpticalElementParams(surface_type=SurfacesTypes.thick_lens,    x=6.5057257992e-03     , y=0                    , z=0                    , t=0                    , p=1e+00 * np.pi        , r_1=7.967931913299999e-03, r_2=7.967931913299999e-03, curvature_sign=1.0, T_c=3.0114515984e-03     , n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=1.76e+00             , alpha_expansion=5.5e-06              , beta_surface_absorption=1e-06                , kappa_conductivity=4.606e+01            , dn_dT=1.17e-05             , nu_poisson_ratio=3e-01                , alpha_volume_absorption=1e-02                , intensity_reflectivity=1e-04                , intensity_transmittance=9.99899e-01          , temperature=np.nan               )),
          OpticalElementParams(surface_type=SurfacesTypes.curved_mirror, x=-5e-03               , y=0                    , z=0                    , t=0                    , p=-1e+00 * np.pi       , r_1=5.0000281233e-03     , r_2=np.nan               , curvature_sign=1.0, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=np.nan               , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=np.nan               , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=np.nan               , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]

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
perturbed_cavity = perturb_cavity(cavity, (0, 3), 0)

# theta_initial_guess, phi_initial_guess = perturbed_cavity.default_initial_angles
phi_initial_guess = np.pi
y_initial_guess = 0
theta_initial_guess = 0
initial_parameters = np.array(
    [x_coarse + x + x_fine, t_coarse + t + t_fine + theta_initial_guess, y_coarse + y + y_fine + y_initial_guess,
     p_coarse + p + p_fine + phi_initial_guess])
_, ray_history = perturbed_cavity.trace_ray_parametric(initial_parameters)
diff = perturbed_cavity.f_roots(initial_parameters)
fig, ax = plt.subplots()
perturbed_cavity.plot(ax=ax, plot_central_line=False)

for i, ray in enumerate(ray_history):
    ray.plot(ax=ax, label=i)

ax.scatter(perturbed_cavity.surfaces[0].origin[0], perturbed_cavity.surfaces[0].origin[1])
ax.scatter(perturbed_cavity.surfaces[3].origin[0], perturbed_cavity.surfaces[3].origin[1])
if center_around_second_center:
    center_x = perturbed_cavity.surfaces[3].origin[0]
    x_lim = 10 ** x_lim
    ax.set_xlim(center_x - x_lim, center_x + x_lim)
ax.set_ylim(-10 ** y_lim, 10 ** y_lim)
ax.legend()
plt.show()
print(
    f"diff = {diff}\ndiff_norm = {np.linalg.norm(diff)}\ninitial_parameters = {initial_parameters[2]:.10e}, {initial_parameters[3]:.10e}")


# %%

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(figsize=(15, 10))
default_phi = cavity.default_initial_angles[1]
phis = np.linspace(-1e-3, 1e-3, 100)# np.array([0, 2.8e-4, 5.785e-4])
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