from cavity import *
import pickle as pkl
from matplotlib.lines import Line2D
from matplotlib import rc

FONT_SIZE_AXIS_LABELS = 14
FONT_SIZE_TITLES = 14
FONT_SIZE_TICKS = 12

def mm_format(value, tick_number):
    return f"{value * 1e3:.2f}"


def cm_format(value, tick_number):
    return f"{value * 1e2:.2f}"


def kW_format(value, tick_number):
    return f"{value * 1e-3:.0f}"


params = [OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'                  , x=-4.999961263669513e-03, y=0                    , z=0                    , theta=0                    , phi=-1e+00 * np.pi       , r_1=5e-03                , r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )),
          OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'                     , x=6.456898770493272e-03, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2.422e-02            , r_2=5.488e-03            , curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03, n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=1.76e+00             , alpha_expansion=5.5e-06              , beta_surface_absorption=1e-06                , kappa_conductivity=4.606e+01            , dn_dT=1.17e-05             , nu_poisson_ratio=3e-01                , alpha_volume_absorption=1e-02                , intensity_reflectivity=1e-04                , intensity_transmittance=9.99899e-01          , temperature=np.nan               )),
          OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'                  , x=3.573055489216874e-01, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2e-01                , r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]


x_2_values = np.concatenate([np.linspace(-6e-5, 0, 20)])  # np.array([0]),
params_temp = params.copy()
params_temp[1].x = x_2_values[-1]  # Change the position of the lens

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
plot_mirror_lens_mirror_cavity_analysis(cavity, CA=5e-3)

# %%
NAs_2, tolerance_matrix_2 = generate_tolerance_of_NA(params,
                                                     parameter_index_for_NA_control=(1, 0),
                                                     arm_index_for_NA=2,
                                                     parameter_values=x_2_values, lambda_0_laser=lambda_0_laser,
                                                     t_is_trivial=True, p_is_trivial=True, return_cavities=False,
                                                     print_progress=True)
tolerance_matrix_2 = np.abs(tolerance_matrix_2)
# %%
x_1 = 2.0000000000e-02
y_1 = 0.0000000000e+00
t_1 = 0.0000000000e+00
p_1 = 0.0000000000e+00
r_1 = 1.00000071e-02
x_3 = 0
y_3 = 0.0000000000e+00
t_3 = 0.0000000000e+00
p_3 = 0.0000000000e+00
lambda_0_laser = 1064e-09
r_3 = r_1
p_3 += np.pi
names = ['Right Mirror', 'Left Mirror']
# INDICES_DICT = {'x': 0, 'y': 1, 'theta': 2, 'phi': 3, 'r': 4, 'n_1': 5, 'w': 6, 'n_2': 7, 'z': 8, 'curvature_sign': 9,
#                 'alpha_thermal_expansion': 10, 'beta_power_absorption': 11, 'kappa_thermal_conductivity': 12,
#                 'dn_dT': 13,
#                 'nu_poisson_ratio': 14, 'alpha_volume_absorption': 15, 'surface_type': 16}
params = np.array([[x_1, y_1, t_1, p_1, r_1, 0, 0, 0, 0, 1, *PHYSICAL_SIZES_DICT['material_properties_ULE'].to_array, 0],
                   # [x_2, y_2, t_lens, p_2, r_2, n_in, w_2, n_out, 0, 0, 1],
                   [x_3, y_3, t_3, p_3, r_3, 0, 0, 0, 0, 1, *PHYSICAL_SIZES_DICT['material_properties_ULE'].to_array, 0]])
ratios = 1 - np.concatenate((np.array([0]), np.logspace(-7, -3.5, 10, endpoint=True)))
x_1_values = x_1 * ratios
params_temp = params.copy()
params_temp[0, 0] = x_1_values[0]

cavity = Cavity.from_params(params=params_temp, standing_wave=True,
                            lambda_0_laser=lambda_0_laser, names=names, t_is_trivial=True, p_is_trivial=True)
font = {'size': 11}
rc('font', **font)
fig, ax = plt.subplots(figsize=(6, 4))
cavity.plot(axis_span=1, ax=ax)
ax.set_xlim(-0.005, 0.3205)
length = x_1 - x_3 + 0.002
ax.set_ylim(-0.006, 0.006)

def cm_format_shifted(value, tick_number):
    return f"{value * 1e2 - 2:.1f}"
ax.yaxis.set_major_formatter(plt.FuncFormatter(cm_format))
ax.xaxis.set_major_formatter(plt.FuncFormatter(cm_format_shifted))
ax.set_ylabel("y [cm]")
ax.set_xlabel("x [cm]")
ax.set_title(
    f"Fabry Perot, NA={cavity.arms[0].mode_parameters.NA[0]:.3f}, length = {np.linalg.norm(cavity.surfaces_ordered[1].center - cavity.surfaces_ordered[0].center) * 100:.0f}cm")

plt.savefig('figures/systems/Fabry-Perot_scaled_x.svg', dpi=300, bbox_inches='tight')
plt.show()
# %%
NAs_1, tolerance_matrix_1 = generate_tolerance_of_NA(params, parameter_index_for_NA_control=(0, 0), arm_index_for_NA=0,
                                                     parameter_values=x_1_values, t_is_trivial=True, p_is_trivial=True,
                                                     return_cavities=False, print_progress=True)
tolerance_matrix_1 = np.abs(tolerance_matrix_1)
# %%
font = {'size': 20}
rc('font', **font)
fig, ax = plt.subplots(figsize=(10, 10))
plt.title("Tilt Tolerance")
ax.plot(NAs_1, tolerance_matrix_1[0, 2, :], label='Fabry-Perot', linestyle=':', color='g')
ax.plot(NAs_2, tolerance_matrix_2[0, 2, :], label='2-arms, right mirror', linestyle='-', color='g')
ax.plot(NAs_2, tolerance_matrix_2[1, 2, :], label='2-arms, lens', linestyle='--', color='b')
ax.plot(NAs_2, tolerance_matrix_2[2, 2, :], label='2-arms, left mirror', linestyle='-', color='orange')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Numerical Aperture')
ax.set_ylabel('Tolerance [rad]')
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig('figures/NA-tolerance/comparison_tilt.svg', dpi=300, bbox_inches='tight')
plt.show()
# %%
fig, ax = plt.subplots(figsize=(10, 10))
plt.title("Axial Displacement Tolerance")
ax.plot(NAs_1, tolerance_matrix_1[0, 0, :], label='Fabry-Perot', linestyle=':', color='g')
ax.plot(NAs_2, tolerance_matrix_2[0, 0, :], label='2-arms, right mirror', linestyle='-', color='g')
ax.plot(NAs_2, tolerance_matrix_2[1, 0, :], label='2-arms, lens', linestyle='--', color='b', linewidth=2.2)
ax.plot(NAs_2, tolerance_matrix_2[2, 0, :], label='2-arms, left mirror', linestyle='-', color='orange')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Numerical Aperture')
ax.set_ylabel('Tolerance [m]')
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig('figures/NA-tolerance/comparison_lateral.svg', dpi=300, bbox_inches='tight')
plt.show()
# %%
fig, ax = plt.subplots(figsize=(10, 10))
plt.title("Transversal Displacement Tolerance")
ax.plot(NAs_1, tolerance_matrix_1[0, 1, :], label='Fabry-Perot', linestyle=':', color='g')
ax.plot(NAs_2, tolerance_matrix_2[0, 1, :], label='2-arms, right mirror', linestyle='-', color='g')
ax.plot(NAs_2, tolerance_matrix_2[1, 1, :], label='2-arms, lens', linestyle='--', color='b', linewidth=2.2)
ax.plot(NAs_2, tolerance_matrix_2[2, 1, :], label='2-arms, left mirror', linestyle='-', color='orange')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Numerical Aperture')
ax.set_ylabel('Tolerance [m]')
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig('figures/NA-tolerance/comparison_transversal.svg', dpi=300, bbox_inches='tight')
plt.show()
# %%
fig, ax = plt.subplots(figsize=(10, 10))
plt.title("Radius Change Tolerance")
ax.plot(NAs_1, tolerance_matrix_1[0, 4, :], label='Fabry-Perot', linestyle=':', color='g')
ax.plot(NAs_2, tolerance_matrix_2[0, 4, :], label='2-arms, right mirror', linestyle='-', color='g')
ax.plot(NAs_2, tolerance_matrix_2[1, 4, :], label='2-arms, lens', linestyle='--', color='b', linewidth=2.2)
ax.plot(NAs_2, tolerance_matrix_2[2, 4, :], label='2-arms, left mirror', linestyle='-', color='orange')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Numerical Aperture')
ax.set_ylabel('Tolerance [m]')
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig('figures/NA-tolerance/comparison_radius.svg', dpi=300, bbox_inches='tight')
plt.show()
