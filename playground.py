from cavity import *

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


params = [OpticalElementParams(name='Small Mirror',      surface_type='curved_mirror'               , x=-5e-03               , y=0                    , z=0                    , theta=0                    , phi=-1e+00 * np.pi       , r_1=5.000038736030386e-03, r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )),
          OpticalElementParams(name='Lens',              surface_type='thick_lens'                  , x=6.50568057190384e-03 , y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=7.968245017582622e-03, r_2=7.968245017582622e-03, curvature_sign=CurvatureSigns.concave, T_c=3.01136114380768e-03 , n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=1.76e+00             , alpha_expansion=5.5e-06              , beta_surface_absorption=1e-06                , kappa_conductivity=4.606e+01            , dn_dT=1.17e-05             , nu_poisson_ratio=3e-01                , alpha_volume_absorption=1e-02                , intensity_reflectivity=1e-04                , intensity_transmittance=9.99899e-01          , temperature=np.nan               )),
          OpticalElementParams(name='Big Mirror',        surface_type='curved_mirror'               , x=3.080113611438077e-01, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=1.505452062957281e-01, r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]


x_left += x_left_fine
x_right += x_right_fine
r_lens_right += r_lens_right_fine
x_span = 10 ** x_span
y_span = 10 ** y_span
# x_2 += x_2_small_perturbation

params[0].x += x_left;  params[0].y += y_left;  params[0].theta += t_left;  params[0].phi += p_left;  params[0].r_1 += r_left
params[1].x += x_lens;  params[1].y += y_lens;  params[1].theta += t_lens;  params[1].phi += p_lens;  params[1].r_1 += r_lens_left; params[1].r_2 += r_lens_right; params[1].T_c += T_c; params[1].n_inside_or_after += n_in
params[2].x += x_right; params[2].y += y_right; params[2].theta += t_right; params[2].phi += p_right; params[2].r_1 += r_right

cavity = Cavity.from_params(params=params, standing_wave=True,
                            lambda_0_laser=LAMBDA_0_LASER, power=power_laser, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=True)


plot_mirror_lens_mirror_cavity_analysis(cavity)
plt.show()
