from cavity_design import *
from tqdm import tqdm

params = [
          OpticalSurfaceParams(name='Laser Optik Mirror'     ,surface_type='curved_mirror'                  , x=-4.336592383947283e-03  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=5e-03                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None), [
          OpticalSurfaceParams(name='aspheric_lens_automatic - flat side',surface_type='flat_refractive_surface'        , x=4.147283409582568e-03   , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , radius=0                       , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1.574e+00               , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='aspheric_lens_automatic - curved side',surface_type='aspheric_surface'               , x=7.247283409582568e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.704137204127337e-03   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.574e+00               , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=np.array([ 3.78495078e-11,  1.84901860e+02,  3.05345512e+06,  7.70343165e+10, 2.08157839e+15, -2.20570527e+20,  1.08097479e+26, -3.70747171e+31, 7.97890251e+36, -1.09308973e+42,  9.11551800e+46, -4.22614376e+51, 8.35809886e+55]))
         ],
          OpticalSurfaceParams(name='End mirror'             ,surface_type='curved_mirror'                  , x=2.243795166403165e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2e-01                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=2.54e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)]
cavity = Cavity.from_params(params=params, standing_wave=True,
                                lambda_0_laser=LAMBDA_0_LASER, power=28e3, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=True, set_central_line=True, set_mode_parameters=True)
cavity.plot()
plt.title(f"NA_short = {cavity.mode_parameters[0].NA[0]:.2e}, NA_long = {cavity.mode_parameters[4].NA[0]:.2e}")
plt.show()

# %%
perturbations_aspheric_lens = np.linspace(-2e-3, 1e-3, 30)
NAs = np.zeros_like(perturbations_aspheric_lens)
short_arm_lengths = np.zeros_like(perturbations_aspheric_lens)
for i, perturbation_value in enumerate(perturbations_aspheric_lens):
    perturbation_pointer = PerturbationPointer(element_index=1, parameter_name=ParamsNames.x, perturbation_value=perturbation_value)
    perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointer)
    short_arm_lengths[i] = perturbed_cavity.central_line[0].length
    NAs[i] = perturbed_cavity.arms[0].mode_parameters.NA[0]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(short_arm_lengths * 1e3, NAs, marker='o')
ax.set_xlabel('Short Arm Length (mm)')
ax.set_ylabel('Short Arm Numerical Aperture')
ax.set_title('Short Arm Numerical Aperture as a Function of Aspheric lens position Perturbation')
ax.grid()
plt.tight_layout()
# plt.savefig(r'figures\NA_as_a_function_of_big_mirror')
plt.show()
# %%
perturbations_large_mirror = np.linspace(-10e-2, 4e-2, 100)

NAs = np.zeros_like(perturbations_large_mirror)
long_arm_lengths = np.zeros_like(perturbations_large_mirror)
for i, perturbation_value in enumerate(perturbations_large_mirror):
    perturbation_pointer = PerturbationPointer(element_index=2, parameter_name=ParamsNames.x, perturbation_value=perturbation_value)
    perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointer)
    NAs[i] = perturbed_cavity.arms[0].mode_parameters.NA[0]
    long_arm_lengths[i] = perturbed_cavity.arms[3].central_line.length

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(long_arm_lengths * 1e2, NAs, marker='o', markersize=2)
ax.set_xlabel('Long arm length (cm)')
ax.set_ylabel('Short Arm Numerical Aperture')
ax.set_title('Short Arm Numerical Aperture as a Function of Large Mirror Position Perturbation')
ax.grid()
plt.tight_layout()
# plt.savefig(r'figures\NA_as_a_function_of_small_mirror')
plt.show()

# %%
n_lens = 40
n_mirror = 80
lens_perturbations = np.linspace(-2e-3, 1e-3, n_lens)
big_mirrors_perturbations = np.linspace(-10e-3, 200e-3, n_mirror)
NAs = np.zeros((n_lens, n_mirror))

# # grid mapping of the NA as a function of perturbations
# for i in tqdm(range(n_lens)):
#     for j in tqdm(range(n_mirror)):
#         perturbation_pointer_1 = PerturbationPointer(element_index=1, parameter_name='x', perturbation_value=lens_perturbations[i])
#         perturbation_pointer_2 = PerturbationPointer(element_index=2, parameter_name='x', perturbation_value=big_mirrors_perturbations[j])
#         cavity_perturbed = perturb_cavity(cavity=cavity, perturbation_pointer=[perturbation_pointer_1, perturbation_pointer_2])
#         NAs[i, j] = cavity_perturbed.mode_parameters[0].NA[0]
# # %%
# # Plot the 2d map of the NAs as a function of the lens perturbations and big mirrors perturbations:
# fig, ax = plt.subplots()
# im = ax.imshow(NAs, extent=(big_mirrors_perturbations[0], big_mirrors_perturbations[-1], lens_perturbations[0], lens_perturbations[-1]), origin='lower', aspect='auto')
# ax.set_xlabel('Big mirror perturbation (m)')
# ax.set_ylabel('Lens perturbation (m)')
# ax.set_title('NA as a function of lens and big mirror perturbations')
# fig.colorbar(im, ax=ax, label='NA')
# plt.show()
# %%
params_lens_unperturbed = params[1]
results = np.zeros((n_lens, n_mirror, 3))
for i in tqdm(range(n_lens), desc='lens perturbations', position=0):
    params_lens_perturbed = copy.deepcopy(params_lens_unperturbed)
    params_lens_perturbed[0].x += lens_perturbations[i]
    params_lens_perturbed[1].x += lens_perturbations[i]
    optical_system_temp = OpticalSystem.from_params(params=[params[0], params_lens_perturbed], t_is_trivial=True, p_is_trivial=True, use_paraxial_ray_tracing=True, lambda_0_laser=LAMBDA_0_LASER)
    for j, NA in enumerate(np.linspace(0.02, 0.15, 80)):
        try:
            cavity_completed = optical_system_to_cavity_completion(optical_system=optical_system_temp, NA=NA, end_mirror_ROC=2e-1)
            short_arm_length = cavity_completed.arms[0].central_line.length
            long_arm_length = cavity_completed.arms[2].central_line.length
            results[i, j, :] = [short_arm_length, long_arm_length, NA]
        except ValueError:
            results[i, j, :] = [np.nan, np.nan, NA]

# %% Plot results

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
index_ranges = [(0, 9), (9, 25), (25, n_lens)]
sc = None
for ax, (start, end) in zip(axes, index_ranges):
    sc = ax.scatter(
        results[start:end, :, 0].ravel() * 1e3,
        results[start:end, :, 1].ravel() * 1e2,
        c=results[start:end, :, 2].ravel(),
        cmap='viridis',
        s=20,
    )
    ax.set_xlabel('Short arm length (mm)')
    ax.set_title(f'lens index {start}:{end}')
axes[0].set_ylabel('Long arm length (cm)')
fig.suptitle('Long arm length vs short arm length colored by NA')
fig.colorbar(sc, ax=axes, label='NA')
plt.show()