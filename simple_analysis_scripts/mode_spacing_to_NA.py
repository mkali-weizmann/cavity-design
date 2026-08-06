import warnings
from copy import deepcopy
from cavity_design import *
from scipy.interpolate import interp1d

# Figure number of the dependencies plot, so a measurement can find it and mark itself on it
# after the (slow) simulation has already run - see mark_mode_spacing().
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
                                        collimation_point: float,
                                        plot_dependencies=True):
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

    if plot_dependencies:
        plot_dependencies_figure(short_arm_lengths, NAs, mode_spacing)
    plt.show(block=False)

    return NAs, mode_spacing


def plot_dependencies_figure(short_arm_lengths, NAs, mode_spacing):
    """Draw the two-panel dependencies figure and return it.

    Left: mode spacing and NA against the small arm's length. Right: NA against mode spacing.
    The simulated curves are kept on the figure (`dependencies_data` / `dependencies_axes`) so a
    measured mode spacing can be marked on it later, without re-running the simulation.
    """
    # Re-running the simulation replaces the old figure rather than piling axes onto it.
    if plt.fignum_exists(DEPENDENCIES_FIGURE_LABEL):
        plt.close(DEPENDENCIES_FIGURE_LABEL)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), num=DEPENDENCIES_FIGURE_LABEL)

    ax_twin = ax[0].twinx()
    ax_twin.plot(short_arm_lengths, NAs, label='NA')
    ax_twin.set_ylabel('NA')
    ax[0].plot(short_arm_lengths, mode_spacing / 1e6, label=r'Mode spacing',
               color='C1')  # use second default color (first is taken by NA)
    ax[0].grid()
    ax[0].set_xlabel("Small arm's length [m]")
    handles1, labels1 = ax[0].get_legend_handles_labels()
    handles2, labels2 = ax_twin.get_legend_handles_labels()
    ax[0].legend(handles1 + handles2, labels1 + labels2)

    ax[1].plot(mode_spacing / 1e6, NAs)
    ax[1].set_xlabel(r'Mode spacing [MHz]')
    ax[1].set_ylabel('NA')
    ax[1].set_ylim(0, 0.22)
    ax[1].grid()

    plt.suptitle('Dependencies')
    fig.tight_layout()
    fig.legend()

    fig.dependencies_data = (np.asarray(short_arm_lengths, dtype=float),
                             np.asarray(NAs, dtype=float),
                             np.asarray(mode_spacing, dtype=float))
    fig.dependencies_axes = (ax[0], ax[1], ax_twin)
    return fig


def short_arm_for_mode_spacing(short_arm_lengths, mode_spacing, mode_spacing_MHz):
    """Invert the simulated curve: the small arm length that yields `mode_spacing_MHz`.

    `mode_spacing` is the simulated array in Hz. Returns None when the value falls outside the
    simulated range (extrapolating a scan this narrow would be meaningless). The scan is monotonic
    in mode spacing over the useful range; the array is sorted here only because np.interp needs an
    increasing x.
    """
    spacing_mhz = np.asarray(mode_spacing, dtype=float) / 1e6
    arms = np.asarray(short_arm_lengths, dtype=float)
    finite = np.isfinite(spacing_mhz) & np.isfinite(arms)
    spacing_mhz, arms = spacing_mhz[finite], arms[finite]
    if spacing_mhz.size == 0 or not spacing_mhz.min() <= mode_spacing_MHz <= spacing_mhz.max():
        return None
    order = np.argsort(spacing_mhz)
    return float(np.interp(mode_spacing_MHz, spacing_mhz[order], arms[order]))


def mark_mode_spacing(mode_spacing_MHz, fig=None, color='r', linestyle='--'):
    """Mark a measured mode spacing on the dependencies figure.

    Right panel: a vertical line at the spacing itself. Left panel: a vertical line at the small
    arm length that produces it. `fig` defaults to the open dependencies figure. Returns that small
    arm length, or None if the figure is missing or the spacing is outside the simulated scan.
    """
    if fig is None:
        if not plt.fignum_exists(DEPENDENCIES_FIGURE_LABEL):
            warnings.warn('no dependencies figure is open - nothing to mark '
                          '(run the simulation with plot_dependencies=True)')
            return None
        fig = plt.figure(DEPENDENCIES_FIGURE_LABEL)
    short_arm_lengths, _NAs, mode_spacing = fig.dependencies_data
    ax_left, ax_right, ax_twin = fig.dependencies_axes
    label = f'Measured: {mode_spacing_MHz:.4g} MHz'

    ax_right.axvline(mode_spacing_MHz, color=color, ls=linestyle, label=label)
    ax_right.legend()

    short_arm = short_arm_for_mode_spacing(short_arm_lengths, mode_spacing, mode_spacing_MHz)
    if short_arm is None:
        warnings.warn(f'{mode_spacing_MHz:.4g} MHz is outside the simulated range '
                      f'- marking the mode-spacing panel only')
    else:
        ax_left.axvline(short_arm, color=color, ls=linestyle, label=label)
        # rebuild the combined legend so the marker joins the two curves of the twinned axes
        handles1, labels1 = ax_left.get_legend_handles_labels()
        handles2, labels2 = ax_twin.get_legend_handles_labels()
        ax_left.legend(handles1 + handles2, labels1 + labels2)

    fig.canvas.draw_idle()
    return short_arm


def generate_lens_position_dependencies_output(short_arm_lengths: Union[np.ndarray, float, tuple],
                                               mid_arm_length: float,
                                               long_arm_length: float,
                                               N_points=100,
                                               plot_cavity=True, plot_spectrum=True, plot_dependencies=True,
                                               elements=DEFAULT_ELEMENTS):
    cavity, collimation_point = build_cavity(elements)
    if isinstance(short_arm_lengths, (int, float)):
        short_arm_lengths = np.linspace(collimation_point - short_arm_lengths, collimation_point + short_arm_lengths, N_points)
    elif isinstance(short_arm_lengths, tuple):
        short_arm_lengths = np.linspace(collimation_point - short_arm_lengths[0], collimation_point + short_arm_lengths[1], N_points)

    if plot_cavity or plot_spectrum:
        nominal_lengths = np.array([short_arm_lengths[len(short_arm_lengths)//2], mid_arm_length, long_arm_length]) if len(
            cavity.elements) == 4 else np.array([collimation_point, long_arm_length])
        cavity.set_arms_lengths(nominal_lengths)
    if plot_cavity:
        cavity.plot()
        plt.show(block=False)
    if plot_spectrum:
        cavity.plot_spectrum(width_over_fsr=0.01)
        plt.show(block=False)

    NAs, mode_spacing = generate_lens_position_dependencies(short_arm_lengths=short_arm_lengths,
                                                            mid_arm_length=mid_arm_length,
                                                            long_arm_length=long_arm_length,
                                                            cavity=cavity,
                                                            collimation_point=collimation_point,
                                                            plot_dependencies=plot_dependencies)
    mode_spacing_interp = interp1d(mode_spacing, NAs, fill_value='extrapolate')
    mode_spacing_over_fsr_interp = lambda x: mode_spacing_interp(x * cavity.free_spectral_range)
    return mode_spacing_interp, mode_spacing_over_fsr_interp


if __name__ == "__main__":
    mode_spacing_interp, mode_spacing_over_fsr_interp = generate_lens_position_dependencies_output(short_arm_lengths=(0.9e-4, 2e-4),
                                                                                                   mid_arm_length=0.015,
                                                                                                   long_arm_length=0.35,
                                                                                                   N_points=100,
                                                                                                   plot_cavity=True,
                                                                                                   plot_spectrum=True,
                                                                                                   plot_dependencies=True
                                                                                                   )
    # The plots above are shown non-blocking; keep them open when run standalone.
    plt.show(block=True)
