from cavity import *
lambda_0_laser = 1064e-9
x_left = 0.0000000000e+00
x_left_fine = 0.0000000000e+00
y_left = 0.0000000000e+00
t_left = 0.0000000000e+00
p_left = 0.0000000000e+00
r_left = 0.0000000000e+00
x_lens = 0.0000000000e+00
y_lens = 0.0000000000e+00
t_lens = 0.0000000000e+00
p_lens = 0.0000000000e+00
r_lens_left = 0.0000000000e+00
r_lens_left_fine = 0.0000000000e+00
r_lens_right = 0.0000000000e+00
r_lens_right_fine = 0.0000000000e+00
T_c = 0
n_in = 0.0000000000e+00
x_right = 0.0000000000e+00
x_right_fine = 0.0000000000e+00
y_right = 0.0000000000e+00
t_right = 0.0000000000e+00
p_right = 0.0000000000e+00
r_right = 0.0000000000e+00
lambda_0_laser = 1.0640000000e-06
power_laser = 5.0000000000e+04
elev = 3.8000000000e+01
azim = 1.6800000000e+02
camera_center = -1
even_axes = False
auto_set_x = True
x_span = -1.0000000000e+00
auto_set_y = True
y_span = -2.9000000000e+00
dim = 2
savefig = False
print_perturbed_params = False
print_tables = False
print_minimal_thickness = False
print_default_parameters = False
r_1 = 0.0000000000e+00

params = [OpticalElementParams(surface_type='curved_mirror', x=-5e-03               , y=0                    , z=0                    , theta=0                    , phi=-1e+00 * np.pi       , r_1=5.000045315676729e-03, r_2=np.nan               , curvature_sign=CurvatureSigns.convex, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )),
          OpticalElementParams(surface_type='thick_lens'   , x=6.456776823267892e-03, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2.424176903520436e-02, r_2=5.487903137228402e-03, curvature_sign=CurvatureSigns.convex, T_c=2.913553646535783e-03, n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=None                 , beta_surface_absorption=None                 , kappa_conductivity=None                 , dn_dT=None                 , nu_poisson_ratio=None                 , alpha_volume_absorption=None                 , intensity_reflectivity=None                 , intensity_transmittance=None                 , temperature=np.nan               )),
          OpticalElementParams(surface_type='curved_mirror', x=3.079135536465358e-01, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=1.504597593390832e-01, r_2=np.nan               , curvature_sign=CurvatureSigns.convex, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]

x_left += x_left_fine
x_right += x_right_fine
r_lens_right += r_lens_right_fine
x_span = 10 ** x_span
y_span = 10 ** y_span
# x_2 += x_2_small_perturbation

params[0].x += x_left; params[0].y += y_left; params[0].theta += t_left; params[0].phi += p_left; params[0].r_1 += r_left
params[1].x += x_lens; params[1].y += y_lens; params[1].theta += t_lens; params[1].phi += p_lens; params[1].r_1 += r_lens_left; params[1].r_2 += r_lens_right; params[1].T_c += T_c; params[1].n_inside_or_after += n_in
params[2].x += x_right; params[2].y += y_right; params[2].theta += t_right; params[2].phi += p_right; params[2].r_1 += r_right

cavity = Cavity.from_params(params=params, standing_wave=True,
                            lambda_0_laser=lambda_0_laser, power=power_laser, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=True)
unheated_cavity = cavity.thermal_transformation()
# %%
if print_perturbed_params:
    print(cavity.to_params)
    unheated_cavity.print_table(names=['Right mirror', 'Lens - right face', 'Left mirror'])
if dim == 2:
    fig, ax = plt.subplots(2, 1, figsize=(16, 12))
else:
    fig = plt.figure()
    ax = [1, 1]
    ax[0] = fig.add_subplot(211, projection='3d')
    ax[0].view_init(elev=elev, azim=azim)
    ax[1] = fig.add_subplot(221, projection='3d')
    ax[1].view_init(elev=elev, azim=azim)

ax[1] = cavity.plot(dim=dim, axis_span=x_span, camera_center=camera_center, ax=ax[1])

title_a = f"short arm NA={cavity.arms[2].mode_parameters.NA[0]:.2e}, short arm length = {np.linalg.norm(cavity.surfaces[2].center - cavity.surfaces[3].center):.2e} [m]"
title_b = f"long arm NA={cavity.arms[0].mode_parameters.NA[0]:.2e}, long arm length = {np.linalg.norm(cavity.surfaces[1].center - cavity.surfaces[0].center):.2e} [m], spot size={2 * cavity.arms[2].mode_parameters_on_surface_1.spot_size[0]:.2e}"
ax[1].set_title(f"{title_a}\n{title_b}", pad=10)
if auto_set_x:
    cavity_length = np.abs(params[0, 0] - params[2, 0])
    ax[1].set_xlim(params[0, 0] - 0.01 * cavity_length, params[2, 0] + 0.01 * cavity_length)
if auto_set_y:
    y_lim = maximal_lens_height(params[1, 4], params[1, 6]) * 1.1
else:
    y_lim = y_span

ax[1].set_ylim(-y_lim, y_lim)

unheated_cavity.plot(dim=dim, ax=ax[0])
ax[0].set_xlim(ax[1].get_xlim())
ax[0].set_ylim(ax[1].get_ylim())

ax[0].set_title(f"unheatet_cavity, short arm NA={unheated_cavity.arms[2].mode_parameters.NA[0]:.2e}")
ax[0].grid()
ax[1].grid()
if savefig:
    plt.savefig(f"figures/systems/{int(time())} {title_a} {title_b}.svg")

if print_minimal_thickness:
    R = np.real(params[1, INDICES_DICT['r']])
    spot_size = cavity.arms[0].mode_parameters[0].spot_size[0]
    h = 3 * spot_size
    d = R * (1 - np.sqrt(1 - h ** 2 / R ** 2))
    alpha = np.arcsin(h / R)
    d_alternative = R * (1 - np.cos(alpha))
    print(
        f"minimal thickness when allowing sharp edges: {2 * d}\nminimal thickness when adding one milimeter of edge thickness: {2 * d + 1e-3}")
plt.show()
print(cavity.arms[0].mode_parameters_on_surfaces[0].spot_size[0])