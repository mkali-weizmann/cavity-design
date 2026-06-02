# %%
from matplotlib import use
use('TkAgg')  # Or 'TkAgg' if Qt5Agg doesn't work
from cavity_design import *

# Output beam optical system: mirror substrate (transmissive) followed by 35mm collimating lens
optical_system = OpticalSystem(
    elements=[LASER_OPTIK_MIRROR_REFRACTIVE, THORLABS_35MM_COLLIMATING_LENS],
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
