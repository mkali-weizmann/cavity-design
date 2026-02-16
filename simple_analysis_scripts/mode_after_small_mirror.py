import matplotlib.pyplot as plt

from cavity import *

w_0_initial = w_0_of_NA(0.16, lambda_laser=LAMBDA_0_LASER)
mode_0 = ModeParameters(center=np.array([0, 0, 0]),
                        k_vector=np.array([1, 0, 0]),
                        w_0=np.array([w_0_initial, w_0_initial]),
                        n=1,
                        lambda_0_laser=LAMBDA_0_LASER,
                        principle_axes=np.array([[0, 1, 0], [0, 0, 1]]),)

pos_mirror_inner = mode_0.z_of_R(5e-3, output_type=float)
# dummy_surface = FlatSurface(outwards_normal=np.array([1, 0, 0]), point_on_surface=np.array([0, 0, 0]), name='waist_plane')
initial_mode_parameters = mode_0.local_mode_parameters(z_minus_z_0=0)
initial_ray = Ray(origin=mode_0.center[0, :], k_vector=mode_0.k_vector)
surface_1 = CurvedRefractiveSurface(radius=5e-3, outwards_normal=np.array([1, 0, 0]),
                                    center=np.array([pos_mirror_inner, 0, 0]), n_1=1,
                                    n_2=PHYSICAL_SIZES_DICT['material_properties_fused_silica'].refractive_index,
                                    name='mirrors inner surface')
surface_2 = CurvedRefractiveSurface(radius=5e-3, outwards_normal=np.array([1, 0, 0]),
                                    center=np.array([pos_mirror_inner + 3.45e-3, 0, 0]),
                                    n_1=PHYSICAL_SIZES_DICT['material_properties_fused_silica'].refractive_index, n_2=1,
                                    name='mirrors outer surface')
arms_with_modes = simple_mode_propagator(surfaces=[surface_1, surface_2],
                                         local_mode_parameters_initial=initial_mode_parameters,
                                         ray_initial=initial_ray,
                                         initial_mode_on_first_surface=False)
mode_output = arms_with_modes[-1].mode_parameters

local_mode_parameters_1_cm_after_last = arms_with_modes[-1].local_mode_parameters_on_a_point(arms_with_modes[-1].central_line.origin + 0.01 * arms_with_modes[-1].central_line.k_vector)

list_of_spot_size_lines = []
for arm in arms_with_modes:
    spot_size_lines_separated = generate_spot_size_lines(
        arm.mode_parameters,
        first_point=arm.central_line.origin,
        last_point=arm.central_line.origin + arm.central_line.k_vector * arm.central_line.length,
        principle_axes=arm.mode_principle_axes,
        dim=2,
        plane='xy',
    )
    list_of_spot_size_lines.extend(spot_size_lines_separated)

fig, ax = plt.subplots(figsize=(10, 10))

for line in list_of_spot_size_lines:
    ax.plot(
        line[0, :],
        line[1, :],
        color='red',
        linestyle="--",
        alpha=0.8,
        linewidth=0.5,
    )
for surface in [surface_1, surface_2]:
    surface.plot(ax=ax)

local_mode_parameter_8mm = mode_output.local_mode_parameters(z_minus_z_0=surface_2.center[0] - mode_output.center[0, 0] + 8e-3)
local_mode_parameter_13mm = mode_output.local_mode_parameters(z_minus_z_0=surface_2.center[0] - mode_output.center[0, 0] + 13e-3)

textual_description = (f"Mode in the cavity: {mode_0.NA[0]:.3e}\n"
                       f"Mode after small mirror: NA={mode_output.NA[0]:.2e}\n"
                       f"8mm after the mirror w={local_mode_parameter_8mm.spot_size[0]:.2e}, z-z_0={local_mode_parameter_8mm.z_minus_z_0[0]:.2e}"
                       f"\n13mm after the mirror w={local_mode_parameter_13mm.spot_size[0]:.2e}, z-z_0={local_mode_parameter_13mm.z_minus_z_0[0]:.2e}")
plt.title(textual_description)
plt.xlim(-1e-3, 29e-3)
plt.ylim(-10e-3, 10e-3)
plt.grid()
x8 = surface_2.center[0] - mode_output.center[0, 0] + 9e-3
x13 = surface_2.center[0] - mode_output.center[0, 0] + 14e-3
plt.axvline(x8)
plt.axvline(x13)
y_text = 9e-3
ax.text(x8, y_text, '8 mm from surface_2', rotation=90, va='top', ha='right', fontsize=8, color='blue')
ax.text(x13, y_text, '13 mm from surface_2', rotation=90, va='top', ha='left', fontsize=8, color='blue')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig(r'figures\mode_after_small_mirror.png', dpi=300, bbox_inches='tight')




