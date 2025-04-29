from copy import deepcopy
# import matplotlib
# matplotlib.use("Qt5Agg")  # Or "TkAgg" if Qt is unavailable
from cavity import *
from df_over_fsr_to_NA_ratio_output import generate_lens_position_dependencies, delta_xs_lens_actual, params_actual_cavity


def theoretical_df_FSR(L, R_1, R_2):
    g_1 = 1 - L / R_1
    g_2 = 1 - L / R_2
    if np.sign(g_1) > 0 and np.sign(g_2) > 0:
        sign = 1
    elif np.sign(g_1) < 0 and np.sign(g_2) < 0:
        sign = -1
    df_over_FSR = np.arccos(sign * np.sqrt(g_1 * g_2)) / np.pi
    return df_over_FSR


N = 3000
x_1 = 24e-3 - 1e-5
x_2 = x_1 - 86e-2
x_lens = -26e-3

title_actual_len = 'Actual cavity'
parameters_actual_lens = (params_actual_cavity, delta_xs_lens_actual, title_actual_len)

params_ideal_lens = [
    OpticalElementParams(name='Small Mirror', surface_type=SurfacesTypes.curved_mirror, x=x_1, y=0, z=0, theta=0,
                         phi=-1e+00 * np.pi, r_1=12e-03, r_2=np.nan, curvature_sign=CurvatureSigns.concave, T_c=np.nan,
                         n_inside_or_after=1e+00, n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.31e+00, dn_dT=None,
                                                                nu_poisson_ratio=1.7e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan)),
    OpticalElementParams(name='Lens', surface_type=SurfacesTypes.ideal_lens, x=x_1 + 39e-3, y=0, z=0, theta=0, phi=0,
                         r_1=25.4e-3, r_2=np.nan, curvature_sign=CurvatureSigns.concave, T_c=np.nan,
                         n_inside_or_after=1e+00, n_outside_or_before=1e+00,
                         material_properties=PHYSICAL_SIZES_DICT['material_properties_bk7']),
    OpticalElementParams(name='Big Mirror', surface_type=SurfacesTypes.curved_mirror, x=x_2, y=0, z=0, theta=0, phi=0,
                         r_1=0.5, r_2=np.nan, curvature_sign=CurvatureSigns.concave, T_c=np.nan,
                         n_inside_or_after=1e+00, n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.31e+00, dn_dT=None,
                                                                nu_poisson_ratio=1.7e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan))]
delta_xs_lens_ideal = np.linspace(-8e-4, 0.6e-3, N)
title_ideal_lens = 'Ideal cavity'
parameters_ideal_lens = (params_ideal_lens, delta_xs_lens_ideal, title_ideal_lens)

params_fabry_perot = [
    OpticalElementParams(name='Left Mirror', surface_type='curved_mirror', x=-5e-02, y=0, z=0, theta=0,
                         phi=-1e+00 * np.pi, r_1=5.01e-2, r_2=np.nan, curvature_sign=CurvatureSigns.concave, T_c=np.nan,
                         n_inside_or_after=1e+00, n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.31e+00, dn_dT=None,
                                                                nu_poisson_ratio=1.7e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan)),
    OpticalElementParams(name='Right Mirror', surface_type='curved_mirror', x=5e-2, y=0, z=0, theta=0, phi=0,
                         r_1=5.01e-2, r_2=np.nan, curvature_sign=CurvatureSigns.concave, T_c=np.nan,
                         n_inside_or_after=1e+00, n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.31e+00, dn_dT=None,
                                                                nu_poisson_ratio=1.7e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan))]

delta_xs_lens_fabry_perot = np.linspace(-9e-2, 2.55e-3, N)
title_fabry_perot = 'Fabry-Perot cavity'
parameters_fabry_perot = (params_fabry_perot, delta_xs_lens_fabry_perot, title_fabry_perot)

params, delta_xs_lens, title = parameters_actual_lens

cavity = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=LAMBDA_0_LASER,
                            set_central_line=True,
                            set_mode_parameters=True,
                            set_initial_surface=False,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4,
                            use_paraxial_ray_tracing=True,
                            debug_printing_level=1,
                            )

# cavity.plot()
plot_mirror_lens_mirror_cavity_analysis(cavity, CA=12.5e-3)
plt.ylim(-0.003, 0.003)
plt.xlim(-0.9, 0.05)
# plt.gca().invert_xaxis()
plt.show()

# %%
NAs, df_over_FSR, Ls = generate_lens_position_dependencies(params, delta_xs_lens, plot_dependencies=True)

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(Ls, NAs, label='NA')
ax[0].plot(Ls, df_over_FSR, label=r'$\frac{df}{\text{FSR}}$ - simulation)')  # '
# if len(params) == 2:
#     ax[0].plot(Ls, df_over_FSR_theory, linestyle='--')  # , label=r'$\frac{df}{\text{FSR}}$ - theory'
ax[0].grid()
# ax2 = ax[0].twinx()
# ax2.plot(Ls, delta_xs_lens, label=r'$\Delta{x,\text{lens}}$', color='red')

ax[0].set_xlabel("Small arm's length [m]")
# ax[0].set_xlim(0.048, 0.051)
ax[0].legend()

ax[1].plot(df_over_FSR, NAs)
# ax[1].set_xlabel(r'$\frac{df}{\text{FSR}}$')
ax[1].set_ylabel('NA')
# ax[1].set_xlim(0, 0.12)
ax[1].set_ylim(0, 0.22)
ax[1].grid()
plt.suptitle(title)
fig.tight_layout()
# fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax[0].transAxes)
fig.legend()

plt.show()
# %%
# Predict NA for a given DF/FSR using interpolation:
from scipy.interpolate import interp1d

df_over_FSR_interp = interp1d(df_over_FSR, NAs, fill_value='extrapolate')
DF_over_FSR_example = -0.299
NA_example = df_over_FSR_interp(DF_over_FSR_example)
print(NA_example)

# %%
params_cavity_NA5_confocal = [
    OpticalElementParams(name='Small Mirror', surface_type='curved_mirror', x=2.399e-02, y=-1.355252715606881e-20, z=0,
                         theta=-4.313903376550994e-21 * np.pi, phi=-4.313903376550994e-21 * np.pi, r_1=2.4e-02,
                         r_2=np.nan, curvature_sign=CurvatureSigns.concave, T_c=np.nan, n_inside_or_after=1e+00,
                         n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.31e+00, dn_dT=None,
                                                                nu_poisson_ratio=1.7e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan)),
    OpticalElementParams(name='Lens', surface_type='thick_lens', x=-2.465546000000002e-02, y=-1.355252715606881e-20,
                         z=0, theta=-4.313903376550994e-21 * np.pi, phi=-1e+00 * np.pi, r_1=1e+04, r_2=1.31e-02,
                         curvature_sign=CurvatureSigns.concave, T_c=1.17e-02, n_inside_or_after=1.51e+00,
                         n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=1.507e+00, alpha_expansion=7.1e-06,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.114e+00, dn_dT=None,
                                                                nu_poisson_ratio=2.06e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan)),
    OpticalElementParams(name='Big Mirror', surface_type='curved_mirror', x=-8.3601e-01, y=-1.355252715606881e-20, z=0,
                         theta=-4.313903376550994e-21 * np.pi, phi=-1e+00 * np.pi, r_1=5e-01, r_2=np.nan,
                         curvature_sign=CurvatureSigns.concave, T_c=np.nan, n_inside_or_after=1e+00,
                         n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.31e+00, dn_dT=None,
                                                                nu_poisson_ratio=1.7e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan))]

params_cavity_NA5_concentric = [
    OpticalElementParams(name='Small Mirror', surface_type='curved_mirror', x=2.399e-02, y=-1.355252715606881e-20, z=0,
                         theta=-4.313903376550994e-21 * np.pi, phi=-4.313903376550994e-21 * np.pi, r_1=2.4e-02,
                         r_2=np.nan, curvature_sign=CurvatureSigns.concave, T_c=np.nan, n_inside_or_after=1e+00,
                         n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.31e+00, dn_dT=None,
                                                                nu_poisson_ratio=1.7e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan)),
    OpticalElementParams(name='Lens', surface_type='thick_lens', x=-2.61555772e-02, y=-1.355252715606881e-20, z=0,
                         theta=-4.313903376550994e-21 * np.pi, phi=-1e+00 * np.pi, r_1=1e+04, r_2=1.31e-02,
                         curvature_sign=CurvatureSigns.concave, T_c=1.17e-02, n_inside_or_after=1.51e+00,
                         n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=1.507e+00, alpha_expansion=7.1e-06,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.114e+00, dn_dT=None,
                                                                nu_poisson_ratio=2.06e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan)),
    OpticalElementParams(name='Big Mirror', surface_type='curved_mirror', x=-8.3601e-01, y=-1.355252715606881e-20, z=0,
                         theta=-4.313903376550994e-21 * np.pi, phi=-1e+00 * np.pi, r_1=5e-01, r_2=np.nan,
                         curvature_sign=CurvatureSigns.concave, T_c=np.nan, n_inside_or_after=1e+00,
                         n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.31e+00, dn_dT=None,
                                                                nu_poisson_ratio=1.7e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan))]

cavity_confocal = Cavity.from_params(params=params_cavity_NA5_confocal,
                                     standing_wave=True,
                                     lambda_0_laser=LAMBDA_0_LASER,
                                     set_central_line=True,
                                     set_mode_parameters=True,
                                     set_initial_surface=False,
                                     t_is_trivial=True,
                                     p_is_trivial=True,
                                     power=2e4,
                                     use_paraxial_ray_tracing=True,
                                     debug_printing_level=1,
                                     )

cavity_concentric = Cavity.from_params(params=params_cavity_NA5_concentric,
                                       standing_wave=True,
                                       lambda_0_laser=LAMBDA_0_LASER,
                                       set_central_line=True,
                                       set_mode_parameters=True,
                                       set_initial_surface=False,
                                       t_is_trivial=True,
                                       p_is_trivial=True,
                                       power=2e4,
                                       use_paraxial_ray_tracing=True,
                                       debug_printing_level=1,
                                       )
plot_mirror_lens_mirror_cavity_analysis(cavity_confocal, CA=12.5e-3)
plt.ylim(-0.003, 0.003)
plt.show()
plot_mirror_lens_mirror_cavity_analysis(cavity_concentric, CA=12.5e-3)
plt.ylim(-0.003, 0.003)
plt.show()
# %%

tolerance_matrix_confocal = cavity_confocal.generate_tolerance_matrix()
tolerance_matrix_concentric = cavity_concentric.generate_tolerance_matrix()

# %%
overlaps_series_confocal = cavity_confocal.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix_confocal[:, :]),
                                                                   shift_size=50,
                                                                   )
overlaps_series_concentric = cavity_concentric.generate_overlap_series(
    shifts=2 * np.abs(tolerance_matrix_concentric[:, :]),
    shift_size=50,
)

# # %%
cavity_confocal.generate_overlaps_graphs(overlaps_series=overlaps_series_confocal,
                                         tolerance_matrix=tolerance_matrix_confocal[:, :],
                                         arm_index_for_NA=0)
plt.show()
cavity_concentric.generate_overlaps_graphs(overlaps_series=overlaps_series_concentric,
                                           tolerance_matrix=tolerance_matrix_concentric[:, :],
                                           arm_index_for_NA=0)
plt.show()
