# %%
from cavity_design import *

optical_system = OpticalSystem(elements=[LASER_OPTIK_MIRROR, EDMUND_4p03MM_ASPHERIC_SPHERICAL_VERSION],
                               use_paraxial_ray_tracing=True, p_is_trivial=True, t_is_trivial=True, lambda_0_laser=LAMBDA_0_LASER)


# %%
# This formula comes from find_short_arm_length_for_collimation.nb
short_arm_length = optical_system[0].radius + back_focal_length_of_lens_object(lens_object=optical_system[1])
optical_system.place_element(element=optical_system[1], position=short_arm_length * RIGHT,
                             recalculate_optic=True, reference_center=optical_system[0])
optical_system_reversed = optical_system.inverse

mode_beginning = optical_system.surfaces[-1].center[0]+1e-2
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
print(f"Short arm length: {optical_system.surfaces[1].center[0] - optical_system.surfaces[0].center[0]}")

ax = optical_system_combined.plot()
right_most_plotting_x = 0.2
points = [np.array([right_most_plotting_x, 0, 0])] + [surface.center for surface in optical_system_combined.surfaces] + [np.array([right_most_plotting_x, 0, 0])]
for i, mode in enumerate(propagated_mode_list):
    mode.plot(ax=ax, first_point=points[i], last_point=points[i+1], color='red', linestyle='--')
plt.xlim(-0.01, 0.12)
plt.title(f"Short arm length: {(optical_system.surfaces[1].center[0] - optical_system.surfaces[0].center[0])*1000:.3f}mm")
plt.show()