from copy import deepcopy
# import matplotlib
# matplotlib.use("Qt5Agg")  # Or "TkAgg" if Qt is unavailable


from cavity import *


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
params_actual_lens = [
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
    OpticalElementParams(name='Lens', surface_type='thick_lens', x=-2.1877586e-02, y=-1.355252715606881e-20, z=0,
                         theta=-4.313903376550994e-21 * np.pi, phi=-1e+00 * np.pi, r_1=1.77e-02, r_2=1.77e-02,
                         curvature_sign=CurvatureSigns.concave, T_c=4.2e-03, n_inside_or_after=1.45e+00,
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

delta_xs_lens_actual = np.linspace(-0.0009, 0.0005, N)
title_actual_len = 'Actual cavity'
parameters_actual_lens = (params_actual_lens, delta_xs_lens_actual, title_actual_len)

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
                         material_properties=PHYSICAL_SIZES_DICT['thermal_properties_bk7']),
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
modes_decay_rate = 2
fsr = cavity.free_spectral_range
lorentzian_width = 0.01 * fsr
n_base_mode = 10
n_transversal_modes = 5
main_mode_picks_position = np.arange(n_base_mode) * fsr
transversal_modes_picks_positions = np.arange(n_transversal_modes) * cavity.delta_f_frequency_transversal_modes
picks_positions = main_mode_picks_position[:, None] + transversal_modes_picks_positions[None, :]
picks_amplitudes = np.ones_like(picks_positions)
picks_amplitudes = picks_amplitudes * np.exp(- modes_decay_rate * np.arange(1, n_transversal_modes + 1))[None, :]

x_dummy = np.linspace(transversal_modes_picks_positions[-1], fsr * n_base_mode, 1000)


# Lorentzian Function
def lorentzian(x, x0, gamma, A, y0):
    return A * gamma / (np.pi * ((x - x0) ** 2 + gamma ** 2)) + y0


lorentzians = lorentzian(x_dummy[None, None, :], picks_positions[:, :, None], lorentzian_width, picks_amplitudes[:, :, None], 0)
lorentzians = lorentzians.sum(axis=(0, 1))

colors = ['blue', 'orange', 'green', 'red', 'purple']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

def plot_lorentzians(ax, x_dummy, lorentzians, picks_positions, colors, n_transversal_modes):
    ax.plot(x_dummy, lorentzians)
    y_limit = ax.get_ylim()
    for i in range(n_transversal_modes):
        ax.vlines(picks_positions[:, i], ymin=y_limit[0], ymax=y_limit[1], color=colors[i], linestyle='--', linewidth=0.75, label=f'Mode {i + 1}')
    ax.hlines((y_limit[1] + y_limit[0]) / 2, picks_positions[-2, 0], picks_positions[-2, 1], color='black', linestyle='--', linewidth=0.75, label='Same longitudinal modes')
    ax.set_xlim(x_dummy[0], x_dummy[-1])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [a.u.]')
    ax.legend()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

plot_lorentzians(ax1, x_dummy, lorentzians, picks_positions, colors, n_transversal_modes)

plot_lorentzians(ax2, x_dummy, lorentzians, picks_positions, colors, n_transversal_modes)
ax2.set_xlim(x_dummy[-1], x_dummy[0])

plt.tight_layout()
plt.show()
# %%

NAs = np.zeros(N)
df_over_FSR = np.zeros(N)
df_over_FSR_theory = np.zeros(N)
Ls = np.zeros(N)
for i, delta_x_lens in enumerate(delta_xs_lens):
    params_copy = deepcopy(params)
    params_copy[1].x += delta_x_lens
    cavity = Cavity.from_params(params=params_copy,
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

    NAs[i] = cavity.arms[0].mode_parameters.NA[0]
    Ls[i] = np.linalg.norm(params_copy[1].x - params_copy[0].x)
    try:
        df_over_FSR_temp = cavity.delta_f_frequency_transversal_modes / cavity.free_spectral_range
        df_over_FSR_temp = np.mod(df_over_FSR_temp + 0.5, 1) - 0.5
        df_over_FSR[i] = df_over_FSR_temp
        if len(params_copy) == 2:
            df_over_FSR_theory[i] = theoretical_df_FSR(L=Ls[i], R_1=params_copy[0].r_1,
                                                       R_2=params_copy[1].r_1)

    except (TypeError, FloatingPointError):
        df_over_FSR[i] = np.nan
        df_over_FSR_theory[i] = np.nan

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(Ls, NAs, label='NA')
ax[0].plot(Ls, df_over_FSR, )  # label=r'$\frac{df}{\text{FSR}}$ - simulation'
if len(params) == 2:
    ax[0].plot(Ls, df_over_FSR_theory, linestyle='--')  # , label=r'$\frac{df}{\text{FSR}}$ - theory'
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
DF_over_FSR_example = -0.01
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
