from cavity import *

lambda_0_laser = 1064e-9

params_original = [
    OpticalElementParams(name='Small Mirror', surface_type='curved_mirror', x=-4.988973493761732e-03, y=0, z=0, theta=0,
                         phi=-1e+00 * np.pi, r_1=5e-03, r_2=np.nan, curvature_sign=CurvatureSigns.concave, T_c=np.nan,
                         n_inside_or_after=1e+00, n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.31e+00, dn_dT=None,
                                                                nu_poisson_ratio=1.7e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan)),
    OpticalElementParams(name='Lens', surface_type='thick_lens', x=6.387599281689135e-03, y=0, z=0, theta=0, phi=0,
                         r_1=2.422e-02, r_2=5.488e-03, curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03,
                         n_inside_or_after=1.76e+00, n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=1.76e+00, alpha_expansion=5.5e-06,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=4.606e+01, dn_dT=1.17e-05,
                                                                nu_poisson_ratio=3e-01, alpha_volume_absorption=1e-02,
                                                                intensity_reflectivity=1e-04,
                                                                intensity_transmittance=9.99899e-01,
                                                                temperature=np.nan)),
    OpticalElementParams(name='Big Mirror', surface_type='curved_mirror', x=2.199758914379698e-01, y=0, z=0, theta=0,
                         phi=0, r_1=2e-01, r_2=np.nan, curvature_sign=CurvatureSigns.concave, T_c=np.nan,
                         n_inside_or_after=1e+00, n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.31e+00, dn_dT=None,
                                                                nu_poisson_ratio=1.7e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan))
]

add_unheated_cavity = False
copy_input_parameters = True
copy_cavity_parameters = False
print_perturbations = True
power_laser = 5.0000000000e+04
camera_center = -1
auto_set_x = True
x_span = -1.0000000000e+00
auto_set_y = True
y_span = -2.9000000000e+00
element_index_0 = 1
param_name_0 = 'y'
perturbation_value_special_log_0 = 5.4775423410e+00
perturbation_value_special_log_0_fine = 4.8007986940e+00
element_index_1 = 0
param_name_1 = 'x'
perturbation_value_special_log_1 = 1.7763568394e-15
perturbation_value_special_log_1_fine = 1.7763568394e-15



perturbation_value_0 = widget_convenient_exponent(perturbation_value_special_log_0, base=10, scale=-10)
perturbation_value_1 = widget_convenient_exponent(perturbation_value_special_log_1, base=10, scale=-10)

perturbation_value_0_fine = widget_convenient_exponent(perturbation_value_special_log_0_fine, base=10, scale=-10)
perturbation_value_1_fine = widget_convenient_exponent(perturbation_value_special_log_1_fine, base=10, scale=-10)

perturbation_value_0 += perturbation_value_0_fine
perturbation_value_1 += perturbation_value_1_fine

cavity = Cavity.from_params(params=params_original, standing_wave=True,
                            lambda_0_laser=LAMBDA_0_LASER, power=power_laser, p_is_trivial=True, t_is_trivial=True,
                            use_paraxial_ray_tracing=True, set_central_line=True, set_mode_parameters=True)
perturbation_pointers = [
    PerturbationPointer(element_index=element_index_0, parameter_name=param_name_0,
                        perturbation_value=perturbation_value_0),
    # PerturbationPointer(element_index=element_index_1, parameter_name=param_name_1, perturbation_value=perturbation_value_1)
]
perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointers)
fig, ax = plt.subplots(2, 1, figsize=(12, 12))
plot_mirror_lens_mirror_cavity_analysis(perturbed_cavity, add_unheated_cavity=add_unheated_cavity,
                                        auto_set_x=auto_set_x, x_span=x_span, auto_set_y=auto_set_y, y_span=y_span,
                                        camera_center=camera_center, diameters=[7.75e-3, 7.75e-3, 7.75e-3, 0.0254],
                                        ax=ax[0])
ax[0].axvline(x=cavity.mode_parameters[2].center[0, 0], color='black', linestyle='--', alpha=0.8, linewidth=1)
spot_size_lines_original = cavity.generate_spot_size_lines(dim=2, plane='xy')
for line in spot_size_lines_original:
    ax[0].plot(line[0, :], line[1, :], color='green', linestyle='--', alpha=0.8, linewidth=0.5,
               label="perturbed_mode")
plot_2_cavity_perturbation_overlap(cavity=cavity, second_cavity=perturbed_cavity, real_or_abs='abs', ax=ax[1],
                                   arm_index=2),
plt.suptitle(
    f"param_name_0={param_name_0}, {perturbation_value_0=:.3e}, param_name_1={param_name_1}, {perturbation_value_1=:.3e}\n")
fig.tight_layout()
plt.show()

if print_perturbations:
    final_value_0 = getattr(cavity.params[element_index_0], param_name_0)
    final_value_1 = getattr(cavity.params[element_index_1], param_name_1)
    print(f"{perturbation_value_0=:.3e}\n{perturbation_value_1=:.3e}")