# %%
from cavity_design import *
from scipy.interpolate import interp1d
# The catalog lookup is the one mode_spacing_to_NA.py already uses, so a name that works for the mode-spacing
# simulation works here too, and there is one list of legal names in the project rather than two.
from simple_analysis_scripts.mode_spacing_to_NA import (UnknownCavityElement, available_element_names,
                                                        resolve_elements)

# The cavity has two arms: a short one, then the intracavity lens(es), then a long one that ends on the end mirror.
# The mode is defined by its NA in the *long* arm (there it is matched to the end mirror), but the quantity we
# care about is its NA in the *short* arm. So the long-arm mode is propagated in both directions:
#   - leftwards, out through the mirror substrate and the collimating lens, onto the camera;
#   - rightwards, back into the cavity through the intracavity lens(es), into the short arm.
# Nothing below the definitions runs on import: the plots and the prompt sit under `if __name__ == "__main__"`, so
# another library can import make_spot_size_to_NA_interpolator() without opening a window or asking a question.
#
# The parts lists and the four distances below are only DEFAULTS, for running this file standalone. Every one of them
# is a parameter of make_spot_size_to_NA_interpolator(), so the library that owns the measurement (os-lab's
# utilities/media_tools/postprocessing_camera_video.py) states the system it measured - in particular the long arm
# length, which changes between measurements and is asked for there on every run - and nothing here has to be edited.

DEFAULT_LONG_ARM_LENGTH = 0.4  # end mirror surface -> first intracavity lens.
DEFAULT_MID_ARM_LENGTH = 1e-2  # first intracavity lens -> second one; unused when there is only one.
DEFAULT_LENS_DISTANCE = 59e-3  # end mirror surface -> collimating lens (outside the cavity).
DEFAULT_CAMERA_DISTANCE = 0.02  # last outgoing element -> camera sensor.

# The parts, named after the catalog (available_element_names()) and listed in the order the beam meets them.
DEFAULT_OUTGOING_ELEMENTS = ['COASTLINE_20CM_REFRACTIVE', 'NEWPORT_200MM_PLANO_CONVEX']
DEFAULT_INTRACAVITY_ELEMENTS = ['THOLABS_200MM_PLANO_CONVEX_LENS', 'EDMUND_4p5MM_ASPHERIC_83580']

# Figure label of the dependencies plot: re-running the simulation replaces that window instead of opening another one.
DEPENDENCIES_FIGURE_LABEL = 'Camera spot size -> NA'


class UnsuitableEndMirror(UnknownCavityElement):
    """The outgoing system does not start at a curved mirror, so no mode can be matched to it.

    A subclass of UnknownCavityElement because it is the same kind of mistake - a parts list the caller has to fix -
    and callers that already fail loudly on a bad element name then fail loudly on this too.
    """


def build_systems(outgoing_elements=DEFAULT_OUTGOING_ELEMENTS,
                  intracavity_elements=DEFAULT_INTRACAVITY_ELEMENTS,
                  flip_intracavity: bool = True):
    """Build the two optical systems from catalog names, and publish them at module level.

    `outgoing_elements` is the output beam's path, in the order it meets them: the cavity end mirror in TRANSMISSION
    (the ..._REFRACTIVE entry of the catalog) first, then whatever stands between it and the camera. That first
    surface is what the long-arm mode is matched to, and it is the origin every distance in this file is measured
    from, so the list cannot start with anything else.
    `intracavity_elements` is the way back into the cavity, from the long arm towards the short arm: one lens, or two
    with `mid_arm_length` between them. They are `.flip()`ed (turn off with `flip_intracavity=False`) because the
    catalog orientations belong to the real cavity, where these lenses sit on the *other* side of the end mirror;
    here the layout is mirrored (mirror on the left, short arm on the right), so a plano-convex lens catalogued with
    its flat face towards the mid arm would otherwise present it to the long arm instead.

    Sets `outgoing_system`, `intracavity_system` and `NA_LONG_ARM_MIN`; returns the two systems. The elements still
    have to be positioned afterwards - place_outgoing_lenses() and place_intracavity_lenses() do that.
    """
    global outgoing_system, intracavity_system, NA_LONG_ARM_MIN
    # resolve_elements() hands out deep copies, because place_element moves an element in place and the catalog
    # entries are shared singletons; .flip() likewise returns a turned-around copy.
    outgoing_system = OpticalSystem(elements=resolve_elements(outgoing_elements), use_paraxial_ray_tracing=True,
                                    p_is_trivial=True, t_is_trivial=True)
    radius = getattr(outgoing_system[0][0], 'radius', None)  # a flat surface reports inf, not nothing
    if radius is None or not np.isfinite(radius):
        raise UnsuitableEndMirror(
            f"the outgoing system must start at the cavity's curved end mirror in transmission (a ..._REFRACTIVE "
            f"catalog entry); the first surface of {outgoing_elements[0]!r} has radius {radius}, so there is no "
            f"curvature to match a mode to.")
    # Put the mirror so that its cavity-facing surface sits at the origin.
    outgoing_system.place_element(element=outgoing_system[0], position=outgoing_system[0][0].radius * LEFT,
                                  recalculate_optic=len(outgoing_system.elements) == 1)

    elements = resolve_elements(intracavity_elements)
    intracavity_system = OpticalSystem(elements=[element.flip() for element in elements] if flip_intracavity
                                       else elements,
                                       use_paraxial_ray_tracing=True, p_is_trivial=True, t_is_trivial=True)

    # A mode can only be matched to the end mirror while its Rayleigh range fits in R/2; below the corresponding NA
    # (1.84e-3 for R = 200 mm) match_a_mode_to_mirror raises and names this same floor. Ask the library for it rather
    # than re-deriving it, so the two can never drift apart. It follows the mirror, hence the rebuild here.
    NA_LONG_ARM_MIN = NA_of_z_R(z_R=outgoing_system.surfaces[0].radius / 2, lambda_0_laser=LAMBDA_0_LASER)
    return outgoing_system, intracavity_system


NA_LONG_ARM_MAX = 0.01
build_systems()


def gaps_between_elements(system, gaps, gaps_name: str):
    """The distances between `system`'s consecutive elements, from a scalar (all equal) or a sequence.

    A one-element system has no gaps at all, so whatever `gaps_name` was set to is simply not used - said out loud,
    because a mid arm length that quietly does nothing is a distance someone will believe they simulated.
    """
    n_gaps = len(system.elements) - 1
    if n_gaps == 0:
        if gaps is not None:
            print(f"Note: the system holds a single element, so {gaps_name} = {gaps} is not used.")
        return []
    if gaps is None:
        raise ValueError(f"{gaps_name} is needed: the system holds {n_gaps + 1} elements.")
    if np.isscalar(gaps):
        return [float(gaps)] * n_gaps
    gaps = [float(gap) for gap in gaps]
    if len(gaps) != n_gaps:
        raise ValueError(f"{gaps_name} holds {len(gaps)} distances, but {n_gaps + 1} elements need {n_gaps}.")
    return gaps


def place_outgoing_lenses(lens_distance=DEFAULT_LENS_DISTANCE, camera_distance: float = DEFAULT_CAMERA_DISTANCE):
    """Line the outgoing elements up behind the end mirror and put the camera `camera_distance` behind the last one.

    `lens_distance` is the gap between consecutive outgoing elements (a scalar, or one distance per gap); the mirror
    itself stays where build_systems() put it. Records the sensor's place in the module-level `camera_position`,
    which propagate_long_arm_mode() reads.
    """
    global camera_position
    last = len(outgoing_system.elements) - 1
    for i, gap in enumerate(gaps_between_elements(outgoing_system, lens_distance, 'lens_distance'), start=1):
        outgoing_system.place_element(element=outgoing_system[i], position=gap * LEFT, recalculate_optic=(i == last),
                                      reference_center=outgoing_system[i - 1])
    camera_position = outgoing_system.surfaces[-1].center + camera_distance * LEFT


place_outgoing_lenses()


def place_intracavity_lenses(long_arm_length: float, mid_arm_length=DEFAULT_MID_ARM_LENGTH):
    """Put the first intracavity lens at `long_arm_length` from the end mirror, the rest `mid_arm_length` apart.

    `mid_arm_length` is only consulted when there is more than one lens (see gaps_between_elements).
    """
    last = len(intracavity_system.elements) - 1
    intracavity_system.place_element(element=intracavity_system[0], position=long_arm_length * RIGHT,
                                     recalculate_optic=(last == 0), reference_center=outgoing_system.surfaces[0])
    for i, gap in enumerate(gaps_between_elements(intracavity_system, mid_arm_length, 'mid_arm_length'), start=1):
        intracavity_system.place_element(element=intracavity_system[i], position=gap * RIGHT,
                                         recalculate_optic=(i == last), reference_center=intracavity_system[i - 1])


place_intracavity_lenses(DEFAULT_LONG_ARM_LENGTH)


def propagate_long_arm_mode(NA_long_arm: float):
    """Match a mode of the given NA to the cavity's end mirror and propagate it both ways.

    Returns (camera spot size, short-arm NA, outgoing modes, intracavity modes).
    """
    mode_parameters_long_arm = match_a_mode_to_mirror(
        lambda_0_laser=LAMBDA_0_LASER, mirror=outgoing_system.surfaces[0], NA=NA_long_arm,
        mode_going_away_from_mirror=False,
    )  # Its k_vector points LEFT, i.e. out of the cavity.
    modes_outgoing = outgoing_system.propagate_mode_parameters_return_global(
        mode_parameters_before_first_surface=mode_parameters_long_arm)
    camera_spot_size = modes_outgoing[-1].local_mode_parameters_at_a_point(camera_position).spot_size[0]

    # The same mode, now travelling rightwards - into the cavity, through the intracavity lens(es), into the short arm.
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
    # The beam on the entry face of each intracavity element, whatever the parts list holds.
    surface_index = 0
    for element in intracavity_system.elements:
        surface = intracavity_system.surfaces[surface_index]
        beam_radius = modes_intracavity[surface_index].local_mode_parameters_at_a_point(surface.center).spot_size[0]
        print(f"beam radius on the {element.name} = {beam_radius * 1e3:.3f} mm "
              f"(clear radius {surface.diameter / 2 * 1e3:.2f} mm)")
        surface_index += len(getattr(element, 'surfaces', [element]))

    ax = outgoing_system.plot()
    intracavity_system.plot(ax=ax)
    plot_modes(outgoing_system, modes_outgoing, ax, first_point=ORIGIN, last_point=camera_position + 0.02 * LEFT,
               color='red', linestyle='--')
    plot_modes(intracavity_system, modes_intracavity, ax, first_point=ORIGIN,
               last_point=intracavity_system.surfaces[-1].center + 5e-3 * RIGHT, color='blue', linestyle='--')
    plt.xlim(camera_position[0] - 0.05, intracavity_system.surfaces[-1].center[0] + 0.05)
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
                                      mid_arm_length=DEFAULT_MID_ARM_LENGTH,
                                      lens_distance=DEFAULT_LENS_DISTANCE,
                                      camera_distance: float = DEFAULT_CAMERA_DISTANCE,
                                      outgoing_elements=None,
                                      intracavity_elements=None,
                                      NA_long_arm_range: tuple = None,
                                      N_points: int = 200,
                                      measured_spot_sizes_m=(),
                                      measured_labels=(),
                                      plot: bool = False):
    """Scan the long-arm NA at this geometry and return the camera spot size [m] -> short-arm NA function.

    The scan is what makes the mapping: for every long-arm NA the same mode is propagated out to the camera and back
    into the short arm, giving one (camera spot size, short-arm NA) pair. `NA_long_arm_range` defaults to
    (NA_LONG_ARM_MIN, NA_LONG_ARM_MAX) of whichever end mirror is in use, and cannot start below NA_LONG_ARM_MIN -
    the mode does not match the mirror there.
    The parts lists and all four distances are stated by the caller (see the note at the top of this file), so the
    measuring library never has to edit this one; `outgoing_elements` / `intracavity_elements` rebuild the systems
    (see build_systems for the order and orientation they are read in), and `mid_arm_length` is ignored when the
    cavity holds a single intracavity lens. With `plot=True` the whole system is shown in one window - the two
    dependency panels and the optical system underneath - with every spot size in `measured_spot_sizes_m` (labelled
    by `measured_labels`, e.g. ('w_x', 'w_y')) marked on it.
    The elements are left at the geometry that was simulated; the returned callable no longer depends on them.
    """
    if outgoing_elements is not None or intracavity_elements is not None:
        build_systems(outgoing_elements if outgoing_elements is not None else DEFAULT_OUTGOING_ELEMENTS,
                      intracavity_elements if intracavity_elements is not None else DEFAULT_INTRACAVITY_ELEMENTS)
    if NA_long_arm_range is None:  # read after the rebuild: the floor belongs to the mirror that is now in place
        NA_long_arm_range = (NA_LONG_ARM_MIN, NA_LONG_ARM_MAX)
    if NA_long_arm_range[1] <= NA_long_arm_range[0]:
        # Typically a sharper end mirror than the default one: its NA floor climbed above NA_LONG_ARM_MAX, and the
        # scan would run backwards into NAs the mirror cannot hold. Say so here, where the fix is named.
        raise ValueError(
            f"NA_long_arm_range = {tuple(NA_long_arm_range)} is empty: this end mirror does not support a mode below "
            f"NA = {NA_LONG_ARM_MIN:.4g} (its radius is {outgoing_system.surfaces[0].radius:.4g} m). Raise the range's "
            f"upper end above that floor.")
    place_outgoing_lenses(lens_distance, camera_distance)
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
