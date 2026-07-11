# %%
from cavity_design import *

params = [
          OpticalSurfaceParams(name='Laser Optik Mirror'     ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=5e-03                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None), [
          OpticalSurfaceParams(name='aspheric_lens_automatic - flat side',surface_type='flat_refractive_surface'        , x=2.656000074902883e-03   , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , radius=0                       , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1.574e+00               , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='aspheric_lens_automatic - curved side',surface_type='aspheric_surface'               , x=4.856000074902883e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.372679656101668e-03   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.574e+00               , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=np.array([ 3.88300544e-11,  2.10732198e+02,  4.36719402e+06,  1.27960456e+11, 6.83643180e+15, -2.42583052e+21,  9.26071800e+26, -2.45228128e+32, 4.25464777e+37, -4.88779028e+42,  3.54322754e+47, -1.46376839e+52, 2.62859940e+56]))
         ]]


# %%
# This formula comes from find_short_arm_length_for_collimation.nb
short_arm_length = params[1][1].radius / (params[1][0].n_inside_or_after - 1) + params[0].radius - (params[1][1].x - params[1][0].x) / params[1][0].n_inside_or_after #7.735e-3
# %%
T_c = params[1][1].x - params[1][0].x
params[1][0].x = params[0].x + short_arm_length
params[1][1].x = params[0].x + short_arm_length + T_c
optical_system = OpticalSystem.from_params(params=params, t_is_trivial=True, p_is_trivial=True, use_paraxial_ray_tracing=True)
optical_system_reversed = optical_system.inverse

mode_beginning = params[1][1].x+1e-2
left_going_mode = ModeParameters(center=np.array([mode_beginning, 0, 0]), k_vector=LEFT, lambda_0_laser=LAMBDA_0_LASER, w_0=np.array([1e-3, 1e-3]))

optical_system_combined = OpticalSystem(elements=[*optical_system_reversed.elements, optical_system.elements[1]], use_paraxial_ray_tracing=True)

propagated_local_mode_list = optical_system_combined.propagate_mode_parameters(mode_parameters_before_first_surface=left_going_mode)
directions = [LEFT] * len(optical_system.surfaces) + [RIGHT] * len(optical_system.surfaces)
propagated_mode_list: List[ModeParameters] = []
narrowed_local_parameters = [*propagated_local_mode_list[::2],propagated_local_mode_list[-1]]
for i, m in enumerate(narrowed_local_parameters):
    mode_parameters_temp = m.to_mode_parameters(location_of_local_mode_parameter=[*optical_system_combined.surfaces, optical_system_combined.surfaces[-1]][i].center, k_vector=directions[i])
    propagated_mode_list.append(mode_parameters_temp)


output_mode = propagated_mode_list[-1]
output_mode_after_10_cm = output_mode.local_mode_parameters_at_a_point(p=optical_system_combined.surfaces[0].center + 30e-2 * RIGHT)
output_mode_after_20_cm = output_mode.local_mode_parameters_at_a_point(p=optical_system_combined.surfaces[0].center + 30e-2 * RIGHT)
print(f"spot_size_after_10: {output_mode_after_10_cm.spot_size[0]*1e3:.2f} mm")
print(f"spot_size_after_20: {output_mode_after_20_cm.spot_size[0]*1e3:.2f} mm")
print(f"Short arm length: {params[1][0].x - params[0].x}")

ax = optical_system_combined.plot()
right_most_plotting_x = 0.2
points = [np.array([right_most_plotting_x, 0, 0])] + [surface.center for surface in optical_system_combined.surfaces] + [np.array([right_most_plotting_x, 0, 0])]
for i, mode in enumerate(propagated_mode_list):
    mode.plot(ax=ax, first_point=points[i], last_point=points[i+1], color='red', linestyle='--')
plt.xlim(-0.01, 0.12)
plt.title(f"Short arm length: {(params[1][0].x - params[0].x)*1000:.3f}mm")


plt.show()