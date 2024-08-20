from cavity import *

element_index_0 = 0
param_name_0 = 'y'
perturbation_value_special_log_0 = 0
element_index_1 = 2
param_name_1 = 'x'
perturbation_value_special_log_1 = -0
power_laser = 5.0000000000e+04
print_input_parameters = True
print_tables = False
print_default_parameters = False

params_original = [OpticalElementParams(name='Small Mirror',      surface_type='curved_mirror'               , x=-5e-03               , y=0                    , z=0                    , theta=0                    , phi=-1e+00 * np.pi       , r_1=5.000038736030386e-03, r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )),
                   OpticalElementParams(name='Lens',              surface_type='thick_lens'                  , x=6.50568057190384e-03 , y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=7.968245017582622e-03, r_2=7.968245017582622e-03, curvature_sign=CurvatureSigns.concave, T_c=3.01136114380768e-03 , n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=1.76e+00             , alpha_expansion=5.5e-06              , beta_surface_absorption=1e-06                , kappa_conductivity=4.606e+01            , dn_dT=1.17e-05             , nu_poisson_ratio=3e-01                , alpha_volume_absorption=1e-02                , intensity_reflectivity=1e-04                , intensity_transmittance=9.99899e-01          , temperature=np.nan               )),
                   OpticalElementParams(name='Big Mirror',        surface_type='curved_mirror'               , x=3.080113611438077e-01, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=1.505452062957281e-01, r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]

perturbation_value_0 = widget_convenient_exponent(perturbation_value_special_log_0)
perturbation_value_1 = widget_convenient_exponent(perturbation_value_special_log_1)

cavity = Cavity.from_params(params=params_original, standing_wave=True,
                            lambda_0_laser=LAMBDA_0_LASER, power=power_laser, p_is_trivial=True, t_is_trivial=True,
                            use_paraxial_ray_tracing=True)
perturbation_pointers = [PerturbationPointer(element_index=element_index_0, parameter_name=param_name_0,
                                             perturbation_value=perturbation_value_0),
                         PerturbationPointer(element_index=element_index_1, parameter_name=param_name_1,
                                             perturbation_value=perturbation_value_1)]
perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointers)

plot_mirror_lens_mirror_cavity_analysis(perturbed_cavity, add_unheated_cavity=True)
plt.show()


# # %%
# print(f"cavity r_left = {cavity.params[0].r_1:.9e}, unheated_cavity r_left = {unheated_cavity.to_params[0].r_1:.9e}, diff={cavity.params[0].r_1 - unheated_cavity.to_params[0].r_1:.9e}\n"
#       f"cavity lens focal length = {focal_length_of_lens(cavity.params[1].r_1, cavity.params[1].r_2, cavity.params[1].n_inside_or_after, cavity.params[1].T_c):.4e}, unheated_cavity lens focal length = {focal_length_of_lens(unheated_cavity.to_params[1].r_1, unheated_cavity.to_params[2].r_1, unheated_cavity.to_params[1].n_inside_or_after, unheated_cavity.to_params[2].x - unheated_cavity.to_params[2].x):.4e}, diff={focal_length_of_lens(cavity.params[1].r_1, cavity.params[1].r_2, cavity.params[1].n_inside_or_after, cavity.params[1].T_c) - focal_length_of_lens(unheated_cavity.to_params[1].r_1, unheated_cavity.to_params[2].r_1, unheated_cavity.to_params[1].n_inside_or_after, unheated_cavity.to_params[2].x - unheated_cavity.to_params[2].x):.4e}\n"
#       f"cavity r_right = {cavity.params[2].r_1:.9e}, unheated_cavity r_right = {unheated_cavity.to_params[3].r_1:.9e}, diff={cavity.params[2].r_1 - unheated_cavity.to_params[3].r_1:.9e}")