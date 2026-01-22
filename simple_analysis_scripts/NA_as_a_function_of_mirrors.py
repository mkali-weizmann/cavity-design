from cavity import *
from tqdm import tqdm
# params_original = [
#           OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'                  , x=-4.999961263669513e-03  , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  )),
#           OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'                     , x=6.387599281529567e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=5.488e-03               , curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  )),
#           OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'                  , x=4.078081463927018e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2e-01                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  ))
#          ]  # 40cm long arm

# params_original = [
#           OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'                  , x=-4.999961263669513e-03  , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  )),
#           OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'                     , x=6.458249990515623e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=5.488e-03               , curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  )),
#           OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'                  , x=3.565787616476249e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2e-01                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  ))
#          ] # 35cm long arm

params_original = [
          OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'                  , x=-4.999439852741557e-03  , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  )),
          OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'                     , x=6.387599281689135e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=5.488e-03               , curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  )),
          OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'                  , x=3.82610345680381e-01    , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2e-01                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  ))
         ]  # Low NA configuration


cavity = Cavity.from_params(params=params_original, standing_wave=True,
                                lambda_0_laser=LAMBDA_0_LASER, power=28e3, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=True, set_central_line=True, set_mode_parameters=True)

# %%
perturbations_small_mirror = np.linspace(-10e-3, 1.4e-3, 30)
NAs = np.zeros_like(perturbations_small_mirror)
for i, perturbation_value in enumerate(perturbations_small_mirror):
    perturbation_pointer = PerturbationPointer(element_index=2, parameter_name=ParamsNames.x, perturbation_value=perturbation_value)
    perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointer)
    NAs[i] = perturbed_cavity.arms[0].mode_parameters.NA[0]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(perturbations_small_mirror * 1e3, NAs, marker='o')
ax.set_xlabel('Big Mirror Longitudinal Position Perturbation (mm)')
ax.set_ylabel('Short Arm Numerical Aperture')
ax.set_title('Short Arm Numerical Aperture as a Function of Big Mirror Position Perturbation')
ax.grid()
plt.tight_layout()
plt.savefig(r'figures\NA_as_a_function_of_big_mirror')
plt.show()
# %%
perturbations_large_mirror = np.sort(np.concatenate((np.linspace(-1e-6, 9.7e-5, 10), np.linspace(-1e-6, 1e-6, 21), np.linspace(9.7e-5, 1.4e-3, 10))))

NAs = np.zeros_like(perturbations_large_mirror)
for i, perturbation_value in enumerate(perturbations_large_mirror):
    perturbation_pointer = PerturbationPointer(element_index=0, parameter_name=ParamsNames.x, perturbation_value=perturbation_value)
    perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointer)
    NAs[i] = perturbed_cavity.arms[0].mode_parameters.NA[0]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(perturbations_large_mirror * 1e3, NAs, marker='o', markersize=2)
ax.set_xlabel('Small Mirror Longitudinal Position Perturbation (mm)')
ax.set_ylabel('Short Arm Numerical Aperture')
ax.set_title('Short Arm Numerical Aperture as a Function of Small Mirror Position Perturbation')
ax.grid()
plt.tight_layout()
plt.savefig(r'figures\NA_as_a_function_of_small_mirror')
plt.show()

# %%
P_small, P_large = np.meshgrid(perturbations_small_mirror, perturbations_large_mirror, indexing='ij')
P_small /= 300
P_large *= 5
NAs = np.zeros_like(P_small)
x_small_mirror_original = cavity.to_params[0].x
x_big_mirror_original = cavity.to_params[2].x
for i in tqdm(range(P_small.shape[0])):
    for j in range(P_small.shape[1]):
        perturbation_pointer_small = PerturbationPointer(element_index=0, parameter_name=ParamsNames.x, perturbation_value=P_small[i, j])
        perturbation_pointer_large = PerturbationPointer(element_index=2, parameter_name=ParamsNames.x, perturbation_value=P_large[i, j])
        perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=[perturbation_pointer_small, perturbation_pointer_large])
        NAs[i, j] = perturbed_cavity.arms[0].mode_parameters.NA[0]


from matplotlib.colors import LogNorm

# absolute positions for axes (convert to millimeters)
X = (P_small + x_small_mirror_original) * 1e3
Y = (P_large + x_big_mirror_original) * 1e3

# prepare NA data for log scale (avoid non-positive values)
NAs_col = NAs.copy()
positives = NAs_col > 0
if not np.any(positives):
    NAs_col[:] = 1e-12
else:
    min_pos = np.nanmin(NAs_col[positives])
    NAs_col[~positives] = min_pos * 1e-3

norm = LogNorm(vmin=np.nanmin(NAs_col), vmax=np.nanmax(NAs_col))

fig, ax = plt.subplots(figsize=(8, 6))
pcm = ax.pcolormesh(X, Y, NAs_col, shading='auto', cmap='viridis', norm=norm)
ax.set_xlabel('Small Mirror Longitudinal Position (mm)')
ax.set_ylabel('Big Mirror Longitudinal Position (mm)')
ax.set_title('Short Arm NA as a Function of Mirror Longitudinal Positions')
cbar = fig.colorbar(pcm, ax=ax, format='%.1e')
cbar.set_label('Numerical Aperture (log scale)')
plt.tight_layout()
plt.savefig(r'figures\NA_2d_map.png')
plt.show()