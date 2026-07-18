"""Free-space Gaussian mode propagation through an OpticalSystem (no cavity).

Models the cavity's output path: the transmissive small mirror substrate followed by a
collimating aspheric lens. A mode matched to the mirror curvature at a given NA is launched
backwards through the system, and the beam is queried at a camera plane past the lens.
The second part scans the NA and plots the resulting camera spot size.

Runs in a few seconds; smoke-tested by tests/test_skill_examples.py.
Adapted from simple_analysis_scripts/camera_spot_size_per_cavity_NA.py.
"""

from cavity_design import *

# %% Build the one-pass optical system. The catalog mirror sits at ORIGIN; the lens is placed
# 5.5 mm to its left with .to_position() (non-mutating - the catalog object stays pristine).
mirror = LASER_OPTIK_MIRROR_REFRACTIVE
lens = EKSMA_LENS_20MM_ASPHERIC.to_position(mirror.surfaces[1].center + 5.5e-3 * LEFT)
optical_system = OpticalSystem(
    elements=[mirror, lens],
    use_paraxial_ray_tracing=True,
    p_is_trivial=True,
    t_is_trivial=True,
)


def camera_spot_size(NA: float, camera_distance: float = 0.02) -> float:
    """Spot size (m) at a camera `camera_distance` past the last surface, for a mode leaving
    the cavity with numerical aperture `NA` at the small mirror."""
    mode_initial = match_a_mode_to_mirror(
        lambda_0_laser=LAMBDA_0_LASER,
        mirror=mirror.surfaces[0],
        NA=NA,
        mode_going_away_from_mirror=False,
    )
    # One ModeParameters per region (before / between / after surfaces), global coordinates:
    modes = optical_system.propagate_mode_parameters_return_global(
        mode_parameters_before_first_surface=mode_initial
    )
    camera_plane = optical_system.surfaces[-1].center + camera_distance * LEFT
    return modes[-1].local_mode_parameters_at_a_point(camera_plane).spot_size[0]


# %% Single propagation, plotted: elements from system.plot(), mode overlaid segment by segment.
mode_initial = match_a_mode_to_mirror(
    lambda_0_laser=LAMBDA_0_LASER,
    mirror=mirror.surfaces[0],
    NA=0.02,
    mode_going_away_from_mirror=False,
)
modes = optical_system.propagate_mode_parameters_return_global(
    mode_parameters_before_first_surface=mode_initial
)
ax = optical_system.plot()
modes[0].plot(
    first_point=ORIGIN,
    last_point=mirror.surfaces[0].center,
    ax=ax,
    color="red",
    linestyle="--",
)
for i, mode in enumerate(modes[1:-1]):
    mode.plot(
        first_point=optical_system.surfaces[i].center,
        last_point=optical_system.surfaces[i + 1].center,
        ax=ax,
        color="red",
        linestyle="--",
    )
modes[-1].plot(
    first_point=optical_system.surfaces[-1].center,
    last_point=optical_system.surfaces[-1].center + 0.1 * LEFT,
    ax=ax,
    color="red",
    linestyle="--",
)
plt.xlim(-0.1, 0.001)
plt.show()

# %% Scan: camera spot size vs. NA at the mirror.
NAs = np.linspace(0.02, 0.15, 8)
camera_spot_sizes = np.array([camera_spot_size(NA) for NA in NAs])

plt.figure()
plt.plot(NAs, camera_spot_sizes * 1e3, marker="o")
plt.xlabel("NA at the mirror")
plt.ylabel("Spot size at camera (mm)")
plt.title("Spot size at camera as a function of NA at the mirror")
plt.grid()
plt.show()

# %% Self-checks (keep at the end — they make this example a smoke test).
assert np.isfinite(
    camera_spot_sizes
).all(), "propagation produced non-finite spot sizes"
assert (camera_spot_sizes > 0).all(), "spot sizes must be positive"
assert (camera_spot_sizes < 0.1).all(), "spot sizes are suspiciously large (> 10 cm)"
