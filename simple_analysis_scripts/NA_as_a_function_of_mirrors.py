from cavity_design import *
from tqdm import tqdm
params = [
          OpticalSurfaceParams(name='Laser Optik Mirror'     ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=5e-03                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None), [
          OpticalSurfaceParams(name='aspheric_lens_automatic - flat side',surface_type='flat_refractive_surface'        , x=2.785574700608683e-03   , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , radius=0                       , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1.574e+00               , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='aspheric_lens_automatic - curved side',surface_type='aspheric_surface'               , x=5.885574700608683e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.704137204127337e-03   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.574e+00               , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=np.array([ 3.78495078e-11,  1.84901860e+02,  3.05345512e+06,  7.70343165e+10, 2.08157839e+15, -2.20570527e+20,  1.08097479e+26, -3.70747171e+31, 7.97890251e+36, -1.09308973e+42,  9.11551800e+46, -4.22614376e+51, 8.35809886e+55]))
         ], [
          OpticalSurfaceParams(name='spherical_0'            ,surface_type='curved_refractive_surface'      , x=1.088557470060869e-02   , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=2.542632688301439e-01   , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.51e+00                , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='spherical_1'            ,surface_type='curved_refractive_surface'      , x=1.523557470060868e-02   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.542632688301439e-01   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.51e+00                , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)],
          OpticalSurfaceParams(name='End mirror'             ,surface_type='curved_mirror'                  , x=2.86728334228144e-01    , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2e-01                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=np.nan                  , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)]
cavity = Cavity.from_params(params=params, standing_wave=True,
                                lambda_0_laser=LAMBDA_0_LASER, power=28e3, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=True, set_central_line=True, set_mode_parameters=True)
cavity.plot()
plt.title(f"NA_short = {cavity.mode_parameters[0].NA[0]:.2e}, NA_long = {cavity.mode_parameters[4].NA[0]:.2e}")
plt.show()

# %%
perturbations_aspheric_lens = np.linspace(-10e-5, 1.e-5, 30)
NAs = np.zeros_like(perturbations_aspheric_lens)
for i, perturbation_value in enumerate(perturbations_aspheric_lens):
    perturbation_pointer = PerturbationPointer(element_index=1, parameter_name=ParamsNames.x, perturbation_value=perturbation_value)
    perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointer)
    NAs[i] = perturbed_cavity.arms[0].mode_parameters.NA[0]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(perturbations_aspheric_lens * 1e3, NAs, marker='o')
ax.set_xlabel('Aspheric Lens Perturbation (mm)')
ax.set_ylabel('Short Arm Numerical Aperture')
ax.set_title('Short Arm Numerical Aperture as a Function of Aspheric lens position Perturbation')
ax.grid()
plt.tight_layout()
# plt.savefig(r'figures\NA_as_a_function_of_big_mirror')
plt.show()
# %%
perturbations_large_mirror = np.sort(np.concatenate((np.linspace(-1e-2, 1e-2, 10), np.linspace(-1e-6, 1e-6, 21), np.linspace(9.7e-5, 1.4e-3, 10))))

NAs = np.zeros_like(perturbations_large_mirror)
for i, perturbation_value in enumerate(perturbations_large_mirror):
    perturbation_pointer = PerturbationPointer(element_index=3, parameter_name=ParamsNames.x, perturbation_value=perturbation_value)
    perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointer)
    NAs[i] = perturbed_cavity.arms[0].mode_parameters.NA[0]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(perturbations_large_mirror * 1e3, NAs, marker='o', markersize=2)
ax.set_xlabel('Large Mirror Longitudinal Position Perturbation (mm)')
ax.set_ylabel('Short Arm Numerical Aperture')
ax.set_title('Short Arm Numerical Aperture as a Function of Large Mirror Position Perturbation')
ax.grid()
plt.tight_layout()
# plt.savefig(r'figures\NA_as_a_function_of_small_mirror')
plt.show()

# %%
P_small, P_large = np.meshgrid(perturbations_aspheric_lens, perturbations_large_mirror, indexing='ij')
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
# plt.savefig(r'figures\NA_2d_map.png')
plt.show()