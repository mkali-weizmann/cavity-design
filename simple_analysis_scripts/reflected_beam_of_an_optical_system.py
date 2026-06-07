from cavity_design import *

params = [
          OpticalSurfaceParams(name='Laser Optik Mirror'     ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=5e-03                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None), [
          OpticalSurfaceParams(name='aspheric_lens_automatic - flat side',surface_type='flat_refractive_surface'        , x=4.147283409582568e-03   , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , radius=0                       , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1.574e+00               , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='aspheric_lens_automatic - curved side',surface_type='aspheric_surface'               , x=7.247283409582568e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.704137204127337e-03   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.574e+00               , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=np.array([ 3.78495078e-11,  1.84901860e+02,  3.05345512e+06,  7.70343165e+10, 2.08157839e+15, -2.20570527e+20,  1.08097479e+26, -3.70747171e+31, 7.97890251e+36, -1.09308973e+42,  9.11551800e+46, -4.22614376e+51, 8.35809886e+55]))
         ]]
lens_perturbations = -1.405e-3
params[1][0].x += lens_perturbations
params[1][1].x += lens_perturbations
optical_system = OpticalSystem.from_params(params=params, t_is_trivial=True, p_is_trivial=True, use_paraxial_ray_tracing=True)
optical_system_reversed = optical_system.inverse

mode_beginning = 0.1
left_going_mode = ModeParameters(center=np.array([mode_beginning, 0, 0]), k_vector=LEFT, lambda_0_laser=LAMBDA_0_LASER, w_0=np.array([1e-3, 1e-3]))

optical_system_combined = OpticalSystem(elements=[*optical_system_reversed.elements, optical_system.elements[1]])

propagated_local_mode_list = optical_system_combined.propagate_mode_parameters(mode_parameters_before_first_surface=left_going_mode)
directions = [LEFT] * len(optical_system.surfaces) + [RIGHT] * len(optical_system.surfaces)
propagated_mode_list: List[ModeParameters] = []
narrowed_local_parameters = [*propagated_local_mode_list[::2],propagated_local_mode_list[-1]]
for i, m in enumerate(narrowed_local_parameters):
    mode_parameters_temp = m.to_mode_parameters(location_of_local_mode_parameter=[*optical_system_combined.surfaces, optical_system_combined.surfaces[-1]][i].center, k_vector=directions[i])
    propagated_mode_list.append(mode_parameters_temp)


output_mode = propagated_mode_list[-1]
output_mode_after_10_cm = output_mode.local_mode_parameters_at_a_point(p=optical_system_combined.surfaces[0].center + 10e-2 * RIGHT)
output_mode_after_20_cm = output_mode.local_mode_parameters_at_a_point(p=optical_system_combined.surfaces[0].center + 20e-2 * RIGHT)
print(f"spot_size_after_10: {output_mode_after_10_cm.spot_size[0]*1e3:.2f} mm")
print(f"spot_size_after_20: {output_mode_after_20_cm.spot_size[0]*1e3:.2f} mm")
print(f"Short arm length: {params[1][0].x - params[0].x}")

ax = optical_system_combined.plot()
points = [np.array([mode_beginning, 0, 0])] + [surface.center for surface in optical_system_combined.surfaces] + [np.array([mode_beginning, 0, 0])]
for i, mode in enumerate(propagated_mode_list):
    mode.plot(ax=ax, first_point=points[i], last_point=points[i+1], color='red', linestyle='--')
plt.xlim(-0.01, 0.06)
plt.title(f"Short arm length: {(params[1][0].x - params[0].x)*1000:.2f}mm")
plt.show()
# %%
