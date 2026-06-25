# %%
from matplotlib import use
use('TkAgg')  # Or 'TkAgg' if Qt5Agg doesn't work
from cavity_design import *

# Output beam optical system: mirror substrate (transmissive) followed by 35mm collimating lens

lens_distance = 5.5e-3
set_element_position(EKSMA_LENS_20mm_ASPHERIC, LASER_OPTIK_MIRROR_REFRACTIVE.surfaces[1].center + lens_distance * LEFT),
optical_system = OpticalSystem(
    elements=[LASER_OPTIK_MIRROR_REFRACTIVE, EKSMA_LENS_20mm_ASPHERIC],
    use_paraxial_ray_tracing=True,
    p_is_trivial=True,
    t_is_trivial=True,
)


mode_parameters_initial = match_a_mode_to_mirror(lambda_0_laser=LAMBDA_0_LASER, mirror=LASER_OPTIK_MIRROR_REFRACTIVE.surfaces[0], NA=0.02, mode_going_away_from_mirror=False)
mode_parameters = optical_system.propagate_mode_parameters_return_global(mode_parameters_before_first_surface=mode_parameters_initial)
ax = optical_system.plot()
mode_parameters[0].plot(first_point = ORIGIN, last_point=LASER_OPTIK_MIRROR_REFRACTIVE.surfaces[0].center, ax=ax, color='red', linestyle='--')
mode_parameters[-1].plot(first_point = optical_system.surfaces[-1].center, last_point=optical_system.surfaces[-1].center+0.1*LEFT, ax=ax, color='red', linestyle='--')
for i, mode in enumerate(mode_parameters[1:-1]):
    mode.plot(first_point=optical_system.surfaces[i].center, last_point=optical_system.surfaces[i+1].center, ax=ax, color='red', linestyle='--')
plt.xlim(-0.1, 0.001)
plt.show()
print(mode_parameters[-1].center[0, 0])
# %%
NAs = np.linspace(0.02, 0.15, 20)
camera_spot_sizes = np.zeros_like(NAs)
for i, NA in enumerate(NAs):
    mode_parameters_initial = match_a_mode_to_mirror(lambda_0_laser=LAMBDA_0_LASER,
                                                     mirror=LASER_OPTIK_MIRROR_REFRACTIVE.surfaces[0], NA=NA,
                                                     mode_going_away_from_mirror=False)
    mode_parameters = optical_system.propagate_mode_parameters_return_global(
        mode_parameters_before_first_surface=mode_parameters_initial)
    camera_spot_sizes[i] = mode_parameters[-1].local_mode_parameters_at_a_point(optical_system.surfaces[-1].center+0.02*LEFT).spot_size[0]

plt.plot(NAs, camera_spot_sizes*1e3, marker='o')
plt.xlabel('NA at the mirror')
plt.ylabel('Spot size at camera (mm)')
plt.title('Spot size at camera as a function of NA at the mirror')
plt.grid()
plt.savefig('outputs/figures/camera_spot_size_per_cavity_NA.svg')
plt.show()


