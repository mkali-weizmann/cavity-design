from cavity import *

add_unheated_cavity = False
copy_input_parameters = True
copy_cavity_parameters = False
print_perturbations = False
power_laser = 5.0000000000e+04
camera_center = -1
x_span = -6.0000000000e+00
y_span = -7.0000000000e+00
element_index_0 = 0
param_name_0 = 'phi'
perturbation_value_special_log_0 = 4.93
perturbation_value_special_log_0_fine = 0.0000000000e+00
element_index_1 = 0
param_name_1 = 'x'
perturbation_value_special_log_1 = 1.7763568394e-15
perturbation_value_special_log_1_fine = 1.7763568394e-15
# eval_box = 'cavity.mode_parameters[0].w_0[0], perturbed_cavity.central_line[0].k_vector'


params = [OpticalElementParams(name='None'                   ,surface_type='curved_mirror'                  , x=-4.999964994473332e-03+550e-9  , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=np.nan                  , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  )),
          OpticalElementParams(name='None'                   ,surface_type='curved_mirror'                  , x=4.999964994473332e-03-550e-9   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=np.nan                  , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  )),]


perturbation_value_0 = widget_convenient_exponent(perturbation_value_special_log_0, base=10, scale=-10)
perturbation_value_1 = widget_convenient_exponent(perturbation_value_special_log_1, base=10, scale=-10)

perturbation_value_0_fine = widget_convenient_exponent(perturbation_value_special_log_0_fine, base=10, scale=-10)
perturbation_value_1_fine = widget_convenient_exponent(perturbation_value_special_log_1_fine, base=10, scale=-10)

perturbation_value_0 += perturbation_value_0_fine
perturbation_value_1 += perturbation_value_1_fine

cavity = Cavity.from_params(params=params, standing_wave=True,
                            lambda_0_laser=LAMBDA_0_LASER, power=power_laser, p_is_trivial=True, t_is_trivial=True,
                            use_paraxial_ray_tracing=True, set_central_line=True, set_mode_parameters=True)
perturbation_pointers = [
    PerturbationPointer(element_index=element_index_0, parameter_name=param_name_0,
                        perturbation_value=perturbation_value_0),
    PerturbationPointer(element_index=element_index_1, parameter_name=param_name_1,
                        perturbation_value=perturbation_value_1)
]
perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointers)

fig, ax = plt.subplots(3, 1, figsize=(8, 12))
perturbed_cavity.plot(ax=ax[0])
perturbed_cavity.plot(ax=ax[1])
spot_size_lines_original = cavity.generate_spot_size_lines(dim=2, plane='xy')
for line in spot_size_lines_original:
    ax[0].plot(line[0, :], line[1, :], color='green', linestyle='--', alpha=0.8, linewidth=0.5, label="perturbed_mode")
    ax[1].plot(line[0, :], line[1, :], color='green', linestyle='--', alpha=0.8, linewidth=0.5, label="perturbed_mode")

plot_2_cavity_perturbation_overlap(cavity=cavity, second_cavity=perturbed_cavity, real_or_abs='abs', ax=ax[2])
u = np.linalg.norm(perturbed_cavity.physical_surfaces[0].origin - perturbed_cavity.physical_surfaces[1].origin)
NA_of_u = np.sqrt(2 * LAMBDA_0_LASER / np.pi) * (perturbed_cavity.arms[0].central_line.length * u) ** (-1 / 4)
plt.suptitle(
    f"param_name_0={param_name_0}, {perturbation_value_0=:.3e}, param_name_1={param_name_1}, {perturbation_value_1=:.3e}\nNA (target)={perturbed_cavity.mode_parameters[0].NA[0]:.3e}, u (extracted)={u:.3e}, NA (extracted from u) = {NA_of_u:.3e}")
ax[1].set_ylim(-10 ** y_span, 10 ** y_span)
ax[1].set_xlim(-10 ** x_span, 10 ** x_span)
ax[1].scatter([perturbed_cavity.physical_surfaces[0].origin[0], perturbed_cavity.physical_surfaces[1].origin[0]],
              [perturbed_cavity.physical_surfaces[0].origin[1], perturbed_cavity.physical_surfaces[1].origin[1]], s=10)
fig.tight_layout()
plt.show()

print(np.sin(perturbed_cavity.central_line[1].k_vector[1]) / perturbed_cavity.mode_parameters[0].NA[0])

# Tilt angle makes sense:
# perturbed_cavity.central_line[0].k_vector[1], perturbed_cavity.physical_surfaces[0].origin[1], perturbed_cavity.physical_surfaces[0].radius * perturbation_pointers[0].perturbation_value / (cavity.physical_surfaces[0].origin[0] - cavity.physical_surfaces[1].origin[0])