# %%
from cavity_design import *
from scipy.interpolate import interp1d

# The cavity has two arms: a short one, then two lenses, then a long one that ends on the Coastline 20cm mirror.
# The mode is defined by its NA in the *long* arm (there it is matched to the Coastline mirror), but the quantity we
# care about is its NA in the *short* arm. So the long-arm mode is propagated in both directions:
#   - leftwards, out through the Coastline mirror substrate and the Newport collimating lens, onto the camera;
#   - rightwards, back into the cavity through the two intracavity lenses, into the short arm.
# Nothing below the definitions runs on import: the plots and the prompt sit under `if __name__ == "__main__"`, so
# another library can import make_spot_size_to_NA_interpolator() without opening a window or asking a question.
#
# The four distances below are only DEFAULTS, for running this file standalone. Every one of them is a parameter of
# make_spot_size_to_NA_interpolator(), so the library that owns the measurement (os-lab's
# utilities/media_tools/postprocessing_camera_video.py) states the geometry it measured - in particular the long arm
# length, which changes between measurements and is asked for there on every run - and nothing here has to be edited.

DEFAULT_LONG_ARM_LENGTH = 0.4  # Coastline mirror surface -> Thorlabs 200mm lens.
DEFAULT_MID_ARM_LENGTH = 1e-2  # Thorlabs 200mm lens -> Edmund 4.5mm aspheric.
DEFAULT_LENS_DISTANCE = 59e-3  # Coastline mirror surface -> Newport 200mm collimating lens (outside the cavity).
DEFAULT_CAMERA_DISTANCE = 0.02  # Newport lens -> camera sensor.

# Figure label of the dependencies plot: re-running the simulation replaces that window instead of opening another one.
DEPENDENCIES_FIGURE_LABEL = 'Camera spot size -> NA'

# Output beam optical system: mirror substrate (transmissive) followed by the Newport 200mm collimating lens.
# The catalog elements are deep-copied because place_element moves them in place.
outgoing_system = OpticalSystem(
    elements=[copy.deepcopy(COASTLINE_20CM_REFRACTIVE), copy.deepcopy(NEWPORT_200MM_PLANO_CONVEX)],
    use_paraxial_ray_tracing=True,
    p_is_trivial=True,
    t_is_trivial=True,
)
outgoing_system.place_element(element=outgoing_system[0], position=outgoing_system[0][0].radius * LEFT, recalculate_optic=False)

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


def place_outgoing_lens(lens_distance: float = DEFAULT_LENS_DISTANCE,
                        camera_distance: float = DEFAULT_CAMERA_DISTANCE):
    """Put the Newport collimating lens `lens_distance` from the Coastline mirror, the camera `camera_distance` behind
    it, and record where the camera sensor now is (module-level `camera_position`, read by propagate_long_arm_mode)."""
    global camera_position
    outgoing_system.place_element(element=outgoing_system[1], position=lens_distance * LEFT, recalculate_optic=True,
                                  reference_center=outgoing_system[0])
    camera_position = outgoing_system.surfaces[-1].center + camera_distance * LEFT


place_outgoing_lens()

# A mode can only be matched to the Coastline mirror while its Rayleigh range fits in R/2; below the corresponding
# NA (1.84e-3 for R = 200 mm) match_a_mode_to_mirror raises and names this same floor. Ask the library for it rather
# than re-deriving it, so the two can never drift apart.
NA_LONG_ARM_MIN = NA_of_z_R(z_R=outgoing_system.surfaces[0].radius / 2, lambda_0_laser=LAMBDA_0_LASER)
NA_LONG_ARM_MAX = 0.01


def place_intracavity_lenses(long_arm_length: float, mid_arm_length: float = DEFAULT_MID_ARM_LENGTH):
    """Put the two intracavity lenses at `long_arm_length` from the Coastline mirror, `mid_arm_length` apart."""
    intracavity_system.place_element(element=intracavity_system[0], position=long_arm_length * RIGHT,
                                     recalculate_optic=False, reference_center=outgoing_system.surfaces[0])
    intracavity_system.place_element(element=intracavity_system[1], position=mid_arm_length * RIGHT,
                                     recalculate_optic=True, reference_center=intracavity_system[0])


place_intracavity_lenses(DEFAULT_LONG_ARM_LENGTH)


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


def scan_long_arm_NAs(long_arm_length: float, NAs_long_arm: np.ndarray,
                      mid_arm_length: float = DEFAULT_MID_ARM_LENGTH):
    """Place the lenses at `long_arm_length` and scan the long-arm NA.

    Returns (camera spot sizes [m], short-arm NAs), both the same shape as `NAs_long_arm`. The intracavity lenses are
    left at `long_arm_length` - place them back at the working point afterwards if the caller depends on it.
    """
    place_intracavity_lenses(long_arm_length, mid_arm_length)
    camera_spot_sizes = np.zeros_like(NAs_long_arm)
    NAs_short_arm = np.zeros_like(NAs_long_arm)
    for i, NA_long_arm in enumerate(NAs_long_arm):
        camera_spot_sizes[i], NAs_short_arm[i], _, _ = propagate_long_arm_mode(NA_long_arm)
    return camera_spot_sizes, NAs_short_arm


# %%
if __name__ == "__main__":
    from matplotlib import use
    use('Qt5Agg')  # Or 'TkAgg' if Qt5Agg doesn't work

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
if __name__ == "__main__":
    # One curve per long-arm length. The camera spot size depends only on the outgoing system, so moving the
    # intracavity lenses re-maps the same set of camera spot sizes onto a different range of short-arm NAs - the
    # curves differ along x, not along y.
    NAs_long_arm = np.linspace(NA_LONG_ARM_MIN, NA_LONG_ARM_MAX, 50)
    long_arm_lengths = np.linspace(0.30, 0.46, 9)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for long_arm_length_i in long_arm_lengths:
        camera_spot_sizes, NAs_short_arm = scan_long_arm_NAs(long_arm_length_i, NAs_long_arm)
        ax.plot(camera_spot_sizes * 1e3, NAs_short_arm, linewidth=2,
                label=f'{long_arm_length_i * 1e2:.1f} cm')
    place_intracavity_lenses(DEFAULT_LONG_ARM_LENGTH)  # restore the working point, so cell 1 stays consistent if re-run

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

# %%
# Camera spot size -> short-arm NA, for one chosen long arm length.
# Mirrors mode_spacing_to_NA.py: scan the simulation once, interpolate inside the scanned support only, and refuse to
# extrapolate. This is the entry point for the library that fits a Gaussian to the camera image: it hands the fitted
# spot size to the returned callable and gets the NA the cavity's short arm had at that moment.

OUT_OF_RANGE_ADVICE = ("Simulation did not produce such a spot size, try widening the NAs_long_arm range, or "
                       "increase N_points for better resolution.")


class SpotSizeOutOfRange(ValueError):
    """A camera spot size the simulated NA scan does not cover."""


def _finite_sorted_by_spot_size(camera_spot_sizes, values):
    """(spot size [mm], values) with the non-finite points dropped, sorted by spot size.

    Sorted because interp1d wants an increasing x. The spot size is monotonic in the long-arm NA over the whole
    scanned range, so each spot size comes from exactly one mode - the map is single valued even though the NA in the
    short arm is not monotonic in it (it turns around close to the NA floor, where z_R = R/2).
    """
    spot_sizes_mm = np.asarray(camera_spot_sizes, dtype=float) * 1e3
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(spot_sizes_mm) & np.isfinite(values)
    spot_sizes_mm, values = spot_sizes_mm[finite], values[finite]
    if spot_sizes_mm.size < 2:
        raise SpotSizeOutOfRange(
            f"the simulation produced only {spot_sizes_mm.size} valid spot size point(s). " + OUT_OF_RANGE_ADVICE)
    order = np.argsort(spot_sizes_mm)
    return spot_sizes_mm[order], values[order]


def _check_within_support(spot_size_mm, support_mm):
    """Raise SpotSizeOutOfRange unless `spot_size_mm` lies inside the simulated support."""
    low, high = support_mm
    values = np.atleast_1d(np.asarray(spot_size_mm, dtype=float))
    if not np.all(np.isfinite(values)) or values.min() < low or values.max() > high:
        shown = (f"{values[0]:.4g}" if values.size == 1
                 else np.array2string(values, precision=4, threshold=8))
        raise SpotSizeOutOfRange(
            f"camera spot size {shown} mm is outside the simulated range "
            f"[{low:.4g}, {high:.4g}] mm. " + OUT_OF_RANGE_ADVICE)


def make_spot_size_to_NA(camera_spot_sizes, NAs_short_arm):
    """Build the camera spot size [m] -> short-arm NA interpolator over the simulated support only.

    Unlike a plain interp1d(..., fill_value='extrapolate'), a spot size outside the scanned range raises
    SpotSizeOutOfRange instead of quietly extrapolating a number nobody simulated. The returned callable carries the
    support as `.support_mm`.
    """
    spot_sizes_mm, NAs = _finite_sorted_by_spot_size(camera_spot_sizes, NAs_short_arm)
    support_mm = (float(spot_sizes_mm[0]), float(spot_sizes_mm[-1]))
    interpolate = interp1d(spot_sizes_mm, NAs)

    def spot_size_to_NA(camera_spot_size_m):
        """Short-arm NA for a camera spot size given in meters (1/e^2 field radius, as everywhere in the library)."""
        spot_size_mm = np.asarray(camera_spot_size_m, dtype=float) * 1e3
        _check_within_support(spot_size_mm, support_mm)
        return interpolate(spot_size_mm)

    spot_size_to_NA.support_mm = support_mm
    return spot_size_to_NA


def long_arm_NA_for_spot_size(camera_spot_sizes, NAs_long_arm, camera_spot_size_m):
    """Invert the simulated curve: the long-arm NA - the scan's parameter - that puts `camera_spot_size_m` on the
    camera. Raises SpotSizeOutOfRange for a spot size the scan never produced."""
    spot_sizes_mm, NAs = _finite_sorted_by_spot_size(camera_spot_sizes, NAs_long_arm)
    spot_size_mm = float(camera_spot_size_m) * 1e3
    _check_within_support(spot_size_mm, (float(spot_sizes_mm[0]), float(spot_sizes_mm[-1])))
    return float(np.interp(spot_size_mm, spot_sizes_mm, NAs))


def plot_dependencies_figure(camera_spot_sizes, NAs_long_arm, NAs_short_arm, long_arm_length,
                             measured_spot_sizes_m=(), measured_labels=(), color='r', linestyle='--'):
    """Draw the dependencies figure and return it.

    Top left: camera spot size and short-arm NA against the long-arm NA, the parameter the scan sweeps. Top right:
    short-arm NA against camera spot size - the mapping the caller is about to use. Underneath, spanning both columns,
    the optical system with the mode drawn through it, at the mean of the measured spot sizes (at the middle of the
    scan when there is no measurement).
    Every measured spot size [m] is marked on both top panels - on the left at the long-arm NA that produces it. A
    measurement outside the scanned range is still drawn (so it is visible how far out it fell), just not converted.
    """
    if plt.fignum_exists(DEPENDENCIES_FIGURE_LABEL):
        plt.close(DEPENDENCIES_FIGURE_LABEL)
    fig = plt.figure(figsize=(12, 9), num=DEPENDENCIES_FIGURE_LABEL)
    grid = fig.add_gridspec(2, 2)
    ax_na_long = fig.add_subplot(grid[0, 0])
    ax_map = fig.add_subplot(grid[0, 1])
    ax_system = fig.add_subplot(grid[1, :])  # the optical system spans both columns underneath

    ax_twin = ax_na_long.twinx()
    ax_twin.plot(NAs_long_arm, NAs_short_arm, label='NA in the short arm')
    ax_twin.set_ylabel('NA in the short arm')
    ax_na_long.plot(NAs_long_arm, camera_spot_sizes * 1e3, color='C1',  # C0 is taken by the short-arm NA
                    label='Spot size at camera')
    ax_na_long.set_xlabel('NA in the long arm')
    ax_na_long.set_ylabel('Spot size at camera [mm]')
    ax_na_long.grid()

    ax_map.plot(camera_spot_sizes * 1e3, NAs_short_arm, linewidth=2)
    ax_map.set_xlabel('Spot size at camera [mm]')
    ax_map.set_ylabel('NA in the short arm')
    ax_map.grid()

    measured_NAs_long_arm = []
    labels = list(measured_labels) + [''] * (len(measured_spot_sizes_m) - len(measured_labels))
    for spot_size_m, name in zip(measured_spot_sizes_m, labels):
        label = f'{name + ": " if name else "Measured: "}{spot_size_m * 1e3:.4g} mm'
        ax_map.axvline(spot_size_m * 1e3, color=color, ls=linestyle, label=label)
        try:
            NA_long_arm = long_arm_NA_for_spot_size(camera_spot_sizes, NAs_long_arm, spot_size_m)
        except SpotSizeOutOfRange:
            continue  # outside the scan: marked on the map panel above, but there is no NA to mark it at here
        measured_NAs_long_arm.append(NA_long_arm)
        ax_na_long.axvline(NA_long_arm, color=color, ls=linestyle, label=label)
    if len(measured_spot_sizes_m):
        ax_map.legend()

    # after the markers, so they join the two curves in the twinned axes' combined legend
    handles1, labels1 = ax_na_long.get_legend_handles_labels()
    handles2, labels2 = ax_twin.get_legend_handles_labels()
    ax_na_long.legend(handles1 + handles2, labels1 + labels2)

    # The system is drawn at the measurement, so what is shown is the mode that was measured.
    NA_shown = float(np.mean(measured_NAs_long_arm)) if measured_NAs_long_arm else float(np.mean(NAs_long_arm))
    _, NA_short_arm_shown, modes_outgoing, modes_intracavity = propagate_long_arm_mode(NA_shown)
    outgoing_system.plot(ax=ax_system)
    intracavity_system.plot(ax=ax_system)
    plot_modes(outgoing_system, modes_outgoing, ax_system, first_point=ORIGIN,
               last_point=camera_position + 0.02 * LEFT, color='red', linestyle='--')
    plot_modes(intracavity_system, modes_intracavity, ax_system, first_point=ORIGIN,
               last_point=intracavity_system.surfaces[-1].center + 5e-3 * RIGHT, color='blue', linestyle='--')
    ax_system.set_xlim(camera_position[0] - 0.05, intracavity_system.surfaces[-1].center[0] + 0.05)
    ax_system.set_ylim(-1e-2, 1e-2)
    ax_system.grid()
    ax_system.set_title(f'The system at NA_long_arm = {NA_shown:.4g} -> NA_short_arm = {NA_short_arm_shown:.4g} '
                        f'(camera on the left, short arm on the right)')

    plt.suptitle(f'Camera spot size -> short-arm NA, long arm = {long_arm_length * 1e2:.1f} cm')
    fig.tight_layout()
    return fig


def make_spot_size_to_NA_interpolator(long_arm_length: float = DEFAULT_LONG_ARM_LENGTH,
                                      mid_arm_length: float = DEFAULT_MID_ARM_LENGTH,
                                      lens_distance: float = DEFAULT_LENS_DISTANCE,
                                      camera_distance: float = DEFAULT_CAMERA_DISTANCE,
                                      NA_long_arm_range: tuple = (NA_LONG_ARM_MIN, NA_LONG_ARM_MAX),
                                      N_points: int = 200,
                                      measured_spot_sizes_m=(),
                                      measured_labels=(),
                                      plot: bool = False):
    """Scan the long-arm NA at this geometry and return the camera spot size [m] -> short-arm NA function.

    The scan is what makes the mapping: for every long-arm NA the same mode is propagated out to the camera and back
    into the short arm, giving one (camera spot size, short-arm NA) pair. `NA_long_arm_range` cannot start below
    NA_LONG_ARM_MIN - the mode does not match the Coastline mirror there.
    All four distances are stated by the caller (see the note at the top of this file), so the measuring library never
    has to edit this one. With `plot=True` the whole system is shown in one window - the two dependency panels and the
    optical system underneath - with every spot size in `measured_spot_sizes_m` (labelled by `measured_labels`, e.g.
    ('w_x', 'w_y')) marked on it.
    The elements are left at the geometry that was simulated; the returned callable no longer depends on them.
    """
    place_outgoing_lens(lens_distance, camera_distance)
    NAs_long_arm = np.linspace(NA_long_arm_range[0], NA_long_arm_range[1], N_points)
    camera_spot_sizes, NAs_short_arm = scan_long_arm_NAs(long_arm_length, NAs_long_arm, mid_arm_length)
    spot_size_to_NA = make_spot_size_to_NA(camera_spot_sizes, NAs_short_arm)
    spot_size_to_NA.long_arm_length = long_arm_length
    spot_size_to_NA.camera_spot_sizes = camera_spot_sizes
    spot_size_to_NA.NAs_long_arm = NAs_long_arm
    spot_size_to_NA.NAs_short_arm = NAs_short_arm

    if plot:
        plot_dependencies_figure(camera_spot_sizes, NAs_long_arm, NAs_short_arm, long_arm_length,
                                 measured_spot_sizes_m=measured_spot_sizes_m, measured_labels=measured_labels)
        plt.show(block=False)
    return spot_size_to_NA


def prompt_for_long_arm_length(default: float = DEFAULT_LONG_ARM_LENGTH):
    """Ask for the long arm length in meters; an empty answer keeps `default`.

    Only for running this file standalone. A library that measures a real cavity knows the arm length it measured (and
    asks its own user for it, as os-lab's postprocessing_camera_video.py does) - it passes it to
    make_spot_size_to_NA_interpolator() instead of being prompted from in here.
    """
    while True:
        answer = input(f"Long arm length [m] (Coastline mirror -> Thorlabs 200mm lens) [{default}]: ").strip()
        if not answer:
            return default
        try:
            return float(answer)
        except ValueError:
            print(f"{answer!r} is not a number, try again.")


if __name__ == "__main__":
    chosen_long_arm_length = prompt_for_long_arm_length()
    spot_size_to_NA = make_spot_size_to_NA_interpolator(long_arm_length=chosen_long_arm_length, plot=True)
    low_mm, high_mm = spot_size_to_NA.support_mm
    print(f"\nLong arm = {chosen_long_arm_length * 1e2:.1f} cm. spot_size_to_NA is defined for camera spot sizes in "
          f"[{low_mm:.4g}, {high_mm:.4g}] mm:")
    for spot_size_mm in np.linspace(low_mm, high_mm, 6):
        print(f"  {spot_size_mm:.4g} mm -> NA = {spot_size_to_NA(spot_size_mm * 1e-3):.4f}")
    place_intracavity_lenses(DEFAULT_LONG_ARM_LENGTH)  # restore the working point, so the cells above stay consistent
    plt.show(block=True)
