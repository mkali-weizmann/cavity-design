from copy import deepcopy
from cavity_design import *
from scipy.interpolate import interp1d

# Figure number of the dependencies plot: re-running the simulation then replaces that window
# instead of opening another one.
DEPENDENCIES_FIGURE_LABEL = 'Dependencies'

# The cavity is not built at import time: callers (in particular the os-lab PicoScope analysis
# scripts, which reach this module through pico_scope/mode_analysis.py) choose the elements, so the
# person doing the measurement never has to edit this file.
DEFAULT_ELEMENTS = ["LASER_OPTIK_MIRROR",
                    "EDMUND_4p5MM_ASPHERIC_83580",
                    "THOLABS_200MM_PLANO_CONVEX_LENS",
                    "COASTLINE_20CM_MIRROR"]


class UnknownCavityElement(ValueError):
    """A cavity element name that is not one of available_element_names()."""


def available_element_names():
    """Every element name resolve_elements() accepts, sorted.

    Both the EXISTING_ELEMENTS_REGISTRY keys and the module-level variable names in
    _existing_elements.py: a few elements are registered under a display name ("Dummy Lens") or not
    registered at all, and the variable name is what a user reads off that file.
    """
    import cavity_design
    names = set(EXISTING_ELEMENTS_REGISTRY)
    names.update(name for name, value in vars(cavity_design).items()
                 if not name.startswith('_') and isinstance(value, (Surface, OpticalSystem)))
    return sorted(names)


def lookup_element(name: str):
    """The catalog element called `name` (registry key or module-level variable name)."""
    if name in EXISTING_ELEMENTS_REGISTRY:
        return EXISTING_ELEMENTS_REGISTRY[name]
    import cavity_design
    element = getattr(cavity_design, name, None)
    if isinstance(element, (Surface, OpticalSystem)):
        return element
    raise UnknownCavityElement(
        f"unknown cavity element {name!r}. Available elements: "
        + ", ".join(available_element_names())
    )


def resolve_elements(elements):
    """Catalog names (or ready-made element objects) -> fresh deep copies.

    The copies matter: Cavity.__init__ only does list(elements) and place_element() mutates in
    place, so handing over the catalog singletons themselves would leave them placed and moved for
    everyone else in the session.
    """
    return [deepcopy(lookup_element(element) if isinstance(element, str) else element)
            for element in elements]


def build_cavity(elements=DEFAULT_ELEMENTS, lambda_0_laser=LAMBDA_0_LASER):
    """Build the cavity from catalog names; return (cavity, collimation_point)."""
    cavity = Cavity(elements=resolve_elements(elements),
                    use_paraxial_ray_tracing=True, p_is_trivial=True, t_is_trivial=True,
                    lambda_0_laser=lambda_0_laser)
    aspheric_BFL = back_focal_length_of_lens_object(lens_object=cavity[1])
    collimation_point = cavity[0].radius + aspheric_BFL
    return cavity, collimation_point

# %%

def generate_lens_position_dependencies(short_arm_lengths: np.ndarray,
                                        mid_arm_length: float,
                                        long_arm_length: float,
                                        cavity,
                                        collimation_point: float):
    NAs = np.zeros_like(short_arm_lengths)
    mode_spacing = np.zeros_like(short_arm_lengths)
    # nominal_positions:
    nominal_lengths = np.array([collimation_point, mid_arm_length, long_arm_length]) if len(
        cavity.elements) == 4 else np.array([collimation_point, long_arm_length])
    cavity.set_arms_lengths(nominal_lengths)

    for i, short_arm_length in tqdm(enumerate(short_arm_lengths)):
        cavity.place_element(element=cavity[1], position=short_arm_length * RIGHT, reference_center=cavity[0],
                             recalculate_optic=True)
        NAs[i] = cavity.arms[0].mode_parameters.NA[0]
        try:
            mode_spacing[i] = cavity.mode_spacing_transversal_apparent
        except (TypeError, FloatingPointError):
            mode_spacing[i] = np.nan

    return NAs, mode_spacing


class ModeSpacingOutOfRange(ValueError):
    """A mode spacing the simulated lens scan does not cover."""


OUT_OF_RANGE_ADVICE = ("Simulation did not produce such a mode spacing, try changing the short arm "
                       "lengths range, or increase the N_points for better resolution.")


def _finite_sorted_by_spacing(mode_spacing, values):
    """(mode spacing [MHz], values) with the non-finite points dropped, sorted by mode spacing.

    Sorted because both np.interp and interp1d want an increasing x; the scan is monotonic in mode
    spacing over the useful range anyway. The scan does produce non-finite points (an unstable
    cavity has no mode spacing), and they carry no support.
    """
    spacing_mhz = np.asarray(mode_spacing, dtype=float) / 1e6
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(spacing_mhz) & np.isfinite(values)
    spacing_mhz, values = spacing_mhz[finite], values[finite]
    if spacing_mhz.size < 2:
        raise ModeSpacingOutOfRange(
            f"the simulation produced only {spacing_mhz.size} valid mode spacing point(s). "
            + OUT_OF_RANGE_ADVICE)
    order = np.argsort(spacing_mhz)
    return spacing_mhz[order], values[order]


def _check_within_support(mode_spacing_MHz, support_MHz):
    """Raise ModeSpacingOutOfRange unless `mode_spacing_MHz` lies inside the simulated support."""
    low, high = support_MHz
    values = np.atleast_1d(np.asarray(mode_spacing_MHz, dtype=float))
    if not np.all(np.isfinite(values)) or values.min() < low or values.max() > high:
        shown = (f"{values[0]:.4g}" if values.size == 1
                 else np.array2string(values, precision=4, threshold=8))
        raise ModeSpacingOutOfRange(
            f"mode spacing {shown} MHz is outside the simulated range "
            f"[{low:.4g}, {high:.4g}] MHz. " + OUT_OF_RANGE_ADVICE)


def _make_mode_spacing_interpolator(mode_spacing, values, name):
    """Build a mode spacing [Hz] -> `values` interpolator over the simulated support only.

    Unlike a plain interp1d(..., fill_value='extrapolate'), a value outside the scanned range
    raises ModeSpacingOutOfRange instead of quietly extrapolating a number nobody simulated. The
    returned callable carries the support as `.support_MHz`.
    """
    spacing_mhz, values = _finite_sorted_by_spacing(mode_spacing, values)
    support_MHz = (float(spacing_mhz[0]), float(spacing_mhz[-1]))
    interpolate = interp1d(spacing_mhz, values)

    def mode_spacing_to_value(mode_spacing_Hz):
        mode_spacing_MHz = np.asarray(mode_spacing_Hz, dtype=float) / 1e6
        _check_within_support(mode_spacing_MHz, support_MHz)
        return interpolate(mode_spacing_MHz)

    mode_spacing_to_value.__name__ = name
    mode_spacing_to_value.support_MHz = support_MHz
    return mode_spacing_to_value


def make_mode_spacing_to_na(mode_spacing, NAs):
    """The mode spacing [Hz] -> NA interpolator; see _make_mode_spacing_interpolator."""
    return _make_mode_spacing_interpolator(mode_spacing, NAs, 'mode_spacing_to_na')


def make_mode_spacing_to_short_arm(mode_spacing, short_arm_lengths):
    """The mode spacing [Hz] -> small arm length [m] interpolator.

    The same inversion of the same lens scan that make_mode_spacing_to_na does, read on the other
    axis: it answers "where is the lens, given the spacing I measured", which is the number to act
    on at the bench, while the NA is the number the cavity is characterized by.
    """
    return _make_mode_spacing_interpolator(mode_spacing, short_arm_lengths,
                                           'mode_spacing_to_short_arm')


def short_arm_for_mode_spacing(short_arm_lengths, mode_spacing, mode_spacing_MHz):
    """Invert the simulated curve: the small arm length that yields `mode_spacing_MHz`.

    `mode_spacing` is the simulated array in Hz. Raises ModeSpacingOutOfRange when the value falls
    outside the scanned range - extrapolating a scan this narrow would be meaningless.
    """
    spacing_mhz, arms = _finite_sorted_by_spacing(mode_spacing, short_arm_lengths)
    _check_within_support(mode_spacing_MHz, (float(spacing_mhz[0]), float(spacing_mhz[-1])))
    return float(np.interp(mode_spacing_MHz, spacing_mhz, arms))


def na_for_mode_spacing(mode_spacing, NAs, mode_spacing_MHz):
    """The simulated NA at `mode_spacing_MHz` - the number a measurement is taken for.

    `mode_spacing` is the simulated array in Hz. Raises ModeSpacingOutOfRange when the value falls
    outside the scanned range, exactly as the interpolator does.
    """
    spacing_mhz, nas = _finite_sorted_by_spacing(mode_spacing, NAs)
    _check_within_support(mode_spacing_MHz, (float(spacing_mhz[0]), float(spacing_mhz[-1])))
    return float(np.interp(mode_spacing_MHz, spacing_mhz, nas))


class NAOutOfRange(ValueError):
    """An NA the simulated lens scan never produces."""


def _finite_in_scan_order(short_arm_lengths, NAs):
    """(small arm lengths, NAs) with the non-finite points dropped, still in lens order.

    Deliberately not sorted, unlike _finite_sorted_by_spacing: what makes the NA different from the
    mode spacing is exactly that it is not monotonic along the scan, so the scan's own order is the
    only one in which its branches can be told apart.
    """
    arms = np.asarray(short_arm_lengths, dtype=float)
    nas = np.asarray(NAs, dtype=float)
    finite = np.isfinite(arms) & np.isfinite(nas)
    if finite.sum() < 2:
        raise NAOutOfRange(
            f"the simulation produced only {int(finite.sum())} valid NA point(s). "
            + OUT_OF_RANGE_ADVICE)
    return arms[finite], nas[finite]


def na_minimum(short_arm_lengths, NAs):
    """(small arm length, NA) at the scan's smallest NA - the collimation point in practice."""
    arms, nas = _finite_in_scan_order(short_arm_lengths, NAs)
    smallest = int(np.argmin(nas))
    return float(arms[smallest]), float(nas[smallest])


def short_arms_for_na(short_arm_lengths, NAs, NA):
    """Every small arm length in the scan that produces `NA` - as a rule, two of them.

    The scan cannot be inverted in the NA the way it can in the mode spacing. Moving the lens away
    from the mirror lowers the mode spacing monotonically, so one spacing names one lens position -
    which is what short_arm_for_mode_spacing() relies on. The NA does not behave that way: it falls
    to a minimum near collimation and climbs again beyond it, so one NA names one lens position on
    each branch. Both come back, in lens order, and it takes something else to say which of them the
    cavity is actually on - a mode-spacing measurement, or simply knowing which side of collimation
    the lens was set from.

    Raises NAOutOfRange for an NA the scan never reaches: below its minimum, or above both branches.
    """
    arms, nas = _finite_in_scan_order(short_arm_lengths, NAs)
    offsets = nas - float(NA)
    roots = []
    for i in range(len(offsets) - 1):
        if offsets[i] == 0.0:
            roots.append(float(arms[i]))
        elif offsets[i] * offsets[i + 1] < 0:  # the curve crosses NA between the two samples
            fraction = offsets[i] / (offsets[i] - offsets[i + 1])
            roots.append(float(arms[i] + fraction * (arms[i + 1] - arms[i])))
    if offsets[-1] == 0.0:
        roots.append(float(arms[-1]))
    # A crossing landing exactly on a sample is found twice, once as the root and once as the sign
    # change beside it; a tenth of a scan step apart is the same lens position either way.
    step = abs(float(np.median(np.diff(arms)))) or 1e-12
    roots.sort()
    unique = [root for index, root in enumerate(roots)
              if index == 0 or root - roots[index - 1] > 0.1 * step]
    if not unique:
        arm_at_min, smallest = na_minimum(short_arm_lengths, NAs)
        raise NAOutOfRange(
            f"NA {float(NA):.4g} is outside the NAs the lens scan produces "
            f"[{smallest:.4g} at a small arm length of {arm_at_min * 1e3:.4f} mm, up to "
            f"{nas.max():.4g}]. " + OUT_OF_RANGE_ADVICE)
    return unique


def plot_dependencies_figure(short_arm_lengths, NAs, mode_spacing, cavity=None,
                             measured_mode_spacing_MHz=None, measured_short_arm=None,
                             measured_na=None, color='r', linestyle='--'):
    """Draw the dependencies figure and return it.

    Top left: mode spacing and NA against the small arm's length. Top right: NA against mode
    spacing. Underneath, spanning both columns, the cavity itself (when `cavity` is given - it is
    drawn as it stands, so place it at the geometry you want shown before calling).
    Given a measured mode spacing [MHz], each top panel gets a vertical line marking it - on the
    left at the small arm length that produces it (`measured_short_arm`, computed here when the
    caller does not pass the value it already has) - plus a horizontal line and a marker at the NA
    it maps to (`measured_na`, likewise computed here when not passed), so the figure shows the
    result of the measurement and not only its input. Both numbers, plus the small arm length, are
    spelled out in the marker's legend entry and in the figure's title.
    """
    # Re-running the simulation replaces the old figure rather than opening another one.
    if plt.fignum_exists(DEPENDENCIES_FIGURE_LABEL):
        plt.close(DEPENDENCIES_FIGURE_LABEL)
    fig = plt.figure(figsize=(12, 9), num=DEPENDENCIES_FIGURE_LABEL)
    grid = fig.add_gridspec(2, 2)
    ax_arm = fig.add_subplot(grid[0, 0])
    ax_na = fig.add_subplot(grid[0, 1])
    ax_cavity = fig.add_subplot(grid[1, :])  # the cavity spans both columns underneath

    ax_twin = ax_arm.twinx()
    ax_twin.plot(short_arm_lengths, NAs, label='NA')
    ax_twin.set_ylabel('NA')
    ax_arm.plot(short_arm_lengths, mode_spacing / 1e6, label=r'Mode spacing',
                color='C1')  # use second default color (first is taken by NA)
    ax_arm.grid()
    ax_arm.set_xlabel("Small arm's length [m]")

    ax_na.plot(mode_spacing / 1e6, NAs)
    ax_na.set_xlabel(r'Mode spacing [MHz]')
    ax_na.set_ylabel('NA')
    ax_na.set_ylim(0, 0.22)
    ax_na.grid()

    if measured_mode_spacing_MHz is not None:
        if measured_short_arm is None:
            measured_short_arm = short_arm_for_mode_spacing(short_arm_lengths, mode_spacing,
                                                            measured_mode_spacing_MHz)
        if measured_na is None:
            measured_na = na_for_mode_spacing(mode_spacing, NAs, measured_mode_spacing_MHz)
        # Both readings of the same inversion: the NA the cavity is characterized by, and the
        # small arm length that produces it - the one to act on at the bench. In mm, because the
        # whole scanned span is a fraction of a millimetre wide.
        label = (f'Measured: {measured_mode_spacing_MHz:.4g} MHz -> NA = {measured_na:.4g}, '
                 f'small arm = {measured_short_arm * 1e3:.4f} mm')
        ax_na.axvline(measured_mode_spacing_MHz, color=color, ls=linestyle, label=label)
        ax_na.axhline(measured_na, color=color, ls=linestyle)
        ax_na.plot(measured_mode_spacing_MHz, measured_na, 'o', color=color)
        ax_na.legend()
        # On the left panel the NA is the twinned axis, so the resulting NA is marked there
        ax_arm.axvline(measured_short_arm, color=color, ls=linestyle, label=label)
        ax_twin.axhline(measured_na, color=color, ls=linestyle)
        ax_twin.plot(measured_short_arm, measured_na, 'o', color=color)

    # after the marker, so it joins the two curves in the twinned axes' combined legend
    handles1, labels1 = ax_arm.get_legend_handles_labels()
    handles2, labels2 = ax_twin.get_legend_handles_labels()
    ax_arm.legend(handles1 + handles2, labels1 + labels2)

    if cavity is not None:
        cavity.plot(ax=ax_cavity)
        ax_cavity.set_title('Cavity')
    else:
        ax_cavity.set_axis_off()

    if measured_mode_spacing_MHz is None:
        plt.suptitle('Dependencies')
    else:
        plt.suptitle(f'Dependencies - measured {measured_mode_spacing_MHz:.4g} MHz '
                     f'-> NA = {measured_na:.4g} at a small arm length of '
                     f'{measured_short_arm * 1e3:.4f} mm')
    fig.tight_layout()
    return fig


def generate_lens_position_dependencies_output(short_arm_lengths: Union[np.ndarray, float, tuple],
                                               mid_arm_length: float,
                                               long_arm_length: float,
                                               N_points=100,
                                               plot_system=True,
                                               elements=DEFAULT_ELEMENTS,
                                               measured_mode_spacing_MHz=None):
    """Scan the lens position and return (mode spacing [Hz] -> NA, (df / FSR) -> NA).

    Both interpolators carry a `.short_arm_m` of their own - the same inversion read on the other
    axis, giving the small arm length [m] that produces a given spacing - so a caller that wants
    to know where the lens is does not have to run the scan a second time.

    With plot_system=True the whole system is shown in one window: the two dependency panels and,
    underneath them, the cavity at its nominal geometry. `measured_mode_spacing_MHz` is marked on
    both dependency panels, together with the NA and the small arm length it maps to.
    """
    cavity, collimation_point = build_cavity(elements)
    if isinstance(short_arm_lengths, (int, float)):
        short_arm_lengths = np.linspace(collimation_point - short_arm_lengths, collimation_point + short_arm_lengths, N_points)
    elif isinstance(short_arm_lengths, tuple):
        short_arm_lengths = np.linspace(collimation_point - short_arm_lengths[0], collimation_point + short_arm_lengths[1], N_points)

    NAs, mode_spacing = generate_lens_position_dependencies(short_arm_lengths=short_arm_lengths,
                                                            mid_arm_length=mid_arm_length,
                                                            long_arm_length=long_arm_length,
                                                            cavity=cavity,
                                                            collimation_point=collimation_point)

    # Raises ModeSpacingOutOfRange if the scan never reached the measurement - checked here rather
    # than at plotting time, so the caller hears about it whether or not it asked for a figure.
    measured_short_arm = measured_na = None
    if measured_mode_spacing_MHz is not None:
        measured_short_arm = short_arm_for_mode_spacing(short_arm_lengths, mode_spacing,
                                                        measured_mode_spacing_MHz)
        measured_na = na_for_mode_spacing(mode_spacing, NAs, measured_mode_spacing_MHz)

    # The scan left the lens at its last position. Restore the nominal geometry, then - when there
    # is a measurement - move the lens to the small arm length that reproduces it, so the cavity
    # that gets drawn (and the free_spectral_range read below) is the one that was measured, with
    # the same NA the caller is about to look up.
    nominal_lengths = np.array([short_arm_lengths[len(short_arm_lengths)//2], mid_arm_length, long_arm_length]) if len(
        cavity.elements) == 4 else np.array([collimation_point, long_arm_length])
    cavity.set_arms_lengths(nominal_lengths)
    if measured_short_arm is not None:
        cavity.place_element(element=cavity[1], position=measured_short_arm * RIGHT,
                             reference_center=cavity[0], recalculate_optic=True)

    if plot_system:
        plot_dependencies_figure(short_arm_lengths, NAs, mode_spacing, cavity=cavity,
                                 measured_mode_spacing_MHz=measured_mode_spacing_MHz,
                                 measured_short_arm=measured_short_arm,
                                 measured_na=measured_na)
        plt.show(block=False)

    mode_spacing_interp = make_mode_spacing_to_na(mode_spacing, NAs)
    short_arm_interp = make_mode_spacing_to_short_arm(mode_spacing, short_arm_lengths)
    free_spectral_range = cavity.free_spectral_range

    def mode_spacing_over_fsr_interp(df_over_fsr):
        return mode_spacing_interp(df_over_fsr * free_spectral_range)

    def short_arm_over_fsr_interp(df_over_fsr):
        return short_arm_interp(df_over_fsr * free_spectral_range)

    # The small arm length rides on the NA interpolator rather than becoming a third return value:
    # every caller unpacks a pair (and os-lab's get_na_interpolators a triple), and the two invert
    # the same lens scan, so wherever an NA can be looked up the lens position can be too. Same
    # convention as `.support_MHz` above.
    mode_spacing_interp.short_arm_m = short_arm_interp
    mode_spacing_over_fsr_interp.short_arm_m = short_arm_over_fsr_interp

    # ... and the same scan read from the NA instead, for a measurement that produces one of those
    # and no mode spacing (the camera spot-size route). It is a list, not a value: see
    # short_arms_for_na for why one NA does not name one lens position.
    for interpolator in (mode_spacing_interp, mode_spacing_over_fsr_interp):
        interpolator.short_arms_for_na =             lambda NA: short_arms_for_na(short_arm_lengths, NAs, NA)
        interpolator.na_minimum = na_minimum(short_arm_lengths, NAs)
    return mode_spacing_interp, mode_spacing_over_fsr_interp


if __name__ == "__main__":
    mode_spacing_interp, mode_spacing_over_fsr_interp = generate_lens_position_dependencies_output(short_arm_lengths=(0.9e-4, 2e-4),
                                                                                                   mid_arm_length=0.015,
                                                                                                   long_arm_length=0.35,
                                                                                                   N_points=100,
                                                                                                   plot_system=True,
                                                                                                   )
    # The plots above are shown non-blocking; keep them open when run standalone.
    plt.show(block=True)
