# %%
from matplotlib import use
use('Qt5Agg')  # Or 'TkAgg' if Qt5Agg doesn't work
from cavity_design import *

# Output beam optical system: mirror substrate (transmissive) followed by 35mm collimating lens

lens_distance = 59e-3
optical_system = OpticalSystem(
    elements=[COASTLINE_20CM_REFRACTIVE, NEWPORT_200MM_PLANO_CONVEX],
    use_paraxial_ray_tracing=True,
    p_is_trivial=True,
    t_is_trivial=True,
)
optical_system.place_element(element=optical_system[0], position = optical_system[0][0].radius * LEFT, recalculate_optic=False)
optical_system.place_element(element=optical_system[1], position = lens_distance * LEFT, recalculate_optic=True, reference_center=optical_system[0])

mode_parameters_initial = match_a_mode_to_mirror(lambda_0_laser=LAMBDA_0_LASER, mirror=optical_system.surfaces[0], NA=0.02, mode_going_away_from_mirror=False)
mode_parameters = optical_system.propagate_mode_parameters_return_global(mode_parameters_before_first_surface=mode_parameters_initial)
ax = optical_system.plot()
mode_parameters[0].plot(first_point = ORIGIN, last_point=optical_system.surfaces[0].center, ax=ax, color='red', linestyle='--')
mode_parameters[-1].plot(first_point = optical_system.surfaces[-1].center, last_point=optical_system.surfaces[-1].center+0.2*LEFT, ax=ax, color='red', linestyle='--')
for i, mode in enumerate(mode_parameters[1:-1]):
    mode.plot(first_point=optical_system.surfaces[i].center, last_point=optical_system.surfaces[i+1].center, ax=ax, color='red', linestyle='--')
plt.xlim(-0.5, -0.1)
plt.ylim(-1e-2, 1e-2)
plt.show()
print(mode_parameters[-1].center[0, 0], mode_parameters[-1].NA)
# %%
NAs = np.linspace(0.02, 0.15, 20)
camera_spot_sizes = np.zeros_like(NAs)
for i, NA in enumerate(NAs):
    mode_parameters_initial = match_a_mode_to_mirror(lambda_0_laser=LAMBDA_0_LASER,
                                                     mirror=optical_system.surfaces[0], NA=NA,
                                                     mode_going_away_from_mirror=False)
    mode_parameters = optical_system.propagate_mode_parameters_return_global(
        mode_parameters_before_first_surface=mode_parameters_initial)
    camera_spot_sizes[i] = mode_parameters[-1].local_mode_parameters_at_a_point(optical_system.surfaces[-1].center+0.02*LEFT).spot_size[0]

plt.plot(NAs, camera_spot_sizes*1e3, marker='o')
plt.xlabel('NA at the mirror')
plt.ylabel('Spot size at camera (mm)')
plt.title('Spot size at camera as a function of NA at the mirror')
plt.grid()
save_path = get_obsidian_save_path('camera_spot_size_per_cavity_NA - NEWPORT 200mm lens')
plt.savefig(save_path)
plt.show()


