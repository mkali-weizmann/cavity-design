# %%
from matplotlib import use
use('Qt5Agg')  # Or 'TkAgg' if Qt5Agg doesn't work
from cavity_design import *

# The cavity has two arms: a short one, then two lenses, then a long one that ends on the Coastline 20cm mirror.
# The mode is defined by its NA in the *long* arm (there it is matched to the Coastline mirror), but the quantity we
# care about is its NA in the *short* arm. So the long-arm mode is propagated in both directions:
#   - leftwards, out through the Coastline mirror substrate and the Newport collimating lens, onto the camera;
#   - rightwards, back into the cavity through the two intracavity lenses, into the short arm.

long_arm_length = 0.4  # Coastline mirror surface -> Thorlabs 200mm lens.
mid_arm_length = 1e-2  # Thorlabs 200mm lens -> Edmund 4.5mm aspheric.
lens_distance = 59e-3  # Coastline mirror surface -> Newport 200mm collimating lens (outside the cavity).
camera_distance = 0.02  # Newport lens -> camera sensor.

# Output beam optical system: mirror substrate (transmissive) followed by the Newport 200mm collimating lens.
# The catalog elements are deep-copied because place_element moves them in place.
outgoing_system = OpticalSystem(
    elements=[copy.deepcopy(COASTLINE_20CM_REFRACTIVE), copy.deepcopy(NEWPORT_200MM_PLANO_CONVEX)],
    use_paraxial_ray_tracing=True,
    p_is_trivial=True,
    t_is_trivial=True,
)
outgoing_system.place_element(element=outgoing_system[0], position=outgoing_system[0][0].radius * LEFT, recalculate_optic=False)
outgoing_system.place_element(element=outgoing_system[1], position=lens_distance * LEFT, recalculate_optic=True, reference_center=outgoing_system[0])

# Intracavity optical system, travelled rightwards: the two lenses that separate the long arm from the short arm.
# Both lenses are catalogued for the real cavity, where they sit to the *left* of the Coastline mirror; here the
# layout is mirrored (Coastline on the left, short arm on the right), so each lens has to be turned around with
# .flip() - otherwise the Thorlabs flat face would look at the mid arm instead of the long arm, and the Edmund
# aspheric face would look at the short arm instead of the mid arm.
intracavity_system = OpticalSystem(
    elements=[THOLABS_200MM_PLANO_CONVEX_LENS.flip(), EDMUND_4p5MM_ASPHERIC_83580.flip()],
    use_paraxial_ray_tracing=True,
    p_is_trivial=True,
    t_is_trivial=True,
)
camera_position = outgoing_system.surfaces[-1].center + camera_distance * LEFT


def place_intracavity_lenses(long_arm_length: float):
    """Put the two intracavity lenses at `long_arm_length` from the Coastline mirror, `mid_arm_length` apart."""
    intracavity_system.place_element(element=intracavity_system[0], position=long_arm_length * RIGHT,
                                     recalculate_optic=False, reference_center=outgoing_system.surfaces[0])
    intracavity_system.place_element(element=intracavity_system[1], position=mid_arm_length * RIGHT,
                                     recalculate_optic=True, reference_center=intracavity_system[0])


place_intracavity_lenses(long_arm_length)


def propagate_long_arm_mode(NA_long_arm: float):
    """Match a mode of the given NA to the Coastline mirror and propagate it both ways.

    Returns (camera spot size, short-arm NA, outgoing modes, intracavity modes).
    """
    mode_parameters_long_arm = match_a_mode_to_mirror(
        lambda_0_laser=LAMBDA_0_LASER, mirror=outgoing_system.surfaces[0], NA=NA_long_arm,
        mode_going_away_from_mirror=False,
    )  # Its k_vector points LEFT, i.e. out of the cavity.
    modes_outgoing = outgoing_system.propagate_mode_parameters_return_global(
        mode_parameters_before_first_surface=mode_parameters_long_arm)
    camera_spot_size = modes_outgoing[-1].local_mode_parameters_at_a_point(camera_position).spot_size[0]

    # The same mode, now travelling rightwards - into the cavity, through the two lenses, into the short arm.
    modes_intracavity = intracavity_system.propagate_mode_parameters_return_global(
        mode_parameters_before_first_surface=mode_parameters_long_arm.invert_direction())
    NA_short_arm = modes_intracavity[-1].NA[0]
    return camera_spot_size, NA_short_arm, modes_outgoing, modes_intracavity


def plot_modes(system, modes, ax, first_point, last_point, **kwargs):
    """Overlay the per-region modes of `system` on `ax`, from `first_point` to `last_point`."""
    modes[0].plot(first_point=first_point, last_point=system.surfaces[0].center, ax=ax, **kwargs)
    for i, mode in enumerate(modes[1:-1]):
        mode.plot(first_point=system.surfaces[i].center, last_point=system.surfaces[i + 1].center, ax=ax, **kwargs)
    modes[-1].plot(first_point=system.surfaces[-1].center, last_point=last_point, ax=ax, **kwargs)


NA_long_arm = 0.0019
camera_spot_size, NA_short_arm, modes_outgoing, modes_intracavity = propagate_long_arm_mode(NA_long_arm)
print(f"NA in the long arm = {NA_long_arm:.4f} -> NA in the short arm = {NA_short_arm:.4f}, "
      f"spot size on the camera = {camera_spot_size * 1e3:.3f} mm")
for surface_index, label in ((0, "Thorlabs 200mm lens"), (2, "Edmund 4.5mm aspheric")):
    surface = intracavity_system.surfaces[surface_index]
    beam_radius = modes_intracavity[surface_index].local_mode_parameters_at_a_point(surface.center).spot_size[0]
    print(f"beam radius on the {label} = {beam_radius * 1e3:.3f} mm "
          f"(clear radius {surface.diameter / 2 * 1e3:.2f} mm)")

ax = outgoing_system.plot()
intracavity_system.plot(ax=ax)
plot_modes(outgoing_system, modes_outgoing, ax, first_point=ORIGIN, last_point=outgoing_system.surfaces[-1].center + 0.2 * LEFT,
           color='red', linestyle='--')
plot_modes(intracavity_system, modes_intracavity, ax, first_point=ORIGIN,
           last_point=intracavity_system.surfaces[-1].center + 5e-3 * RIGHT, color='blue', linestyle='--')
plt.xlim(-0.5, 0.25)
plt.ylim(-1e-2, 1e-2)
plt.grid()
plt.show()

# %%
# One curve per long-arm length. The camera spot size depends only on the outgoing system, so moving the intracavity
# lenses re-maps the same set of camera spot sizes onto a different range of short-arm NAs - the curves differ
# along x, not along y.
# A mode can only be matched to the Coastline mirror while its Rayleigh range fits in R/2; below the corresponding
# NA (1.84e-3 for R = 200 mm) match_a_mode_to_mirror raises and names this same floor. Ask the library for it rather
# than re-deriving it, so the two can never drift apart.
NA_long_arm_min = NA_of_z_R(z_R=outgoing_system.surfaces[0].radius / 2, lambda_0_laser=LAMBDA_0_LASER)
NAs_long_arm = np.linspace(NA_long_arm_min, 0.01, 50)
long_arm_lengths = np.linspace(0.30, 0.46, 9)
fig, ax = plt.subplots(figsize=(8, 5.5))
for long_arm_length_i in long_arm_lengths:
    place_intracavity_lenses(long_arm_length_i)
    camera_spot_sizes = np.zeros_like(NAs_long_arm)
    NAs_short_arm = np.zeros_like(NAs_long_arm)
    for i, NA in enumerate(NAs_long_arm):
        camera_spot_sizes[i], NAs_short_arm[i], _, _ = propagate_long_arm_mode(NA)
    ax.plot(camera_spot_sizes * 1e3, NAs_short_arm, linewidth=2,
            label=f'{long_arm_length_i * 1e2:.1f} cm')
place_intracavity_lenses(long_arm_length)  # restore the working point, so cell 1 stays consistent if re-run

ax.set_ylabel('NA in the short arm')
ax.set_xlabel('Spot size at camera (mm)')
ax.set_title('Spot size at camera vs. the NA in the short arm, per long-arm length')
ax.legend(title='Long arm length', frameon=False)
ax.grid(color='0.85', linewidth=0.6)
ax.set_axisbelow(True)
for spine in ('top', 'right'):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
save_path = get_obsidian_save_path('camera_spot_size_per_cavity_NA - NEWPORT 200mm lens')
plt.savefig(save_path)
plt.show()
enable_copy_to_clipboard()