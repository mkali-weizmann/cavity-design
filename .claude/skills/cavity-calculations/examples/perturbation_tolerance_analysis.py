"""Perturbation / tolerance analysis of a mirror-lens-mirror cavity.

Builds a standing-wave cavity (small curved mirror -> aspheric lens -> dummy lens -> large end
mirror), then asks: how far can each element be shifted (x, y) or tilted (phi) before the
perturbed cavity mode's overlap with the unperturbed mode drops to the threshold (default 0.9)?

Outputs:
- an overview plot of the cavity geometry and mode,
- a tolerance dataframe (rows: perturbed parameter, columns: element),
- overlap-vs-shift curves per element/parameter and their summary plot.

Runs in a few seconds; smoke-tested by tests/test_skill_examples.py.
Adapted from characterize-one-cavity-playground.py.
"""

from cavity_design import *

# %% Build the cavity. Catalog elements are placed with .to_position() (returns a positioned
# copy); the end mirror is defined explicitly to show the full SphericalMirror signature.
cavity = Cavity(
    elements=[
        LASER_OPTIK_MIRROR.to_position(np.array([-0.005, 0.0, 0.0])),
        EKSMA_LENS_20MM_ASPHERIC.to_position(
            np.array([0.017623230771841976, 0.0, 0.0])
        ),
        DUMMY_LENS.to_position(np.array([0.03305723077184197, 0.0, 0.0])),
        SphericalMirror(
            name="End mirror",
            radius=0.2,
            outwards_normal=np.array([1.0, 0.0, 0.0]),
            center=np.array([0.4511688875799871, 0.0, 0.0]),
            curvature_sign=-1,
            diameter=0.0254,
            material_properties=MaterialProperties(
                refractive_index=1.45,
                alpha_expansion=5.2e-07,
                beta_surface_absorption=1e-06,
                kappa_conductivity=1.38,
                dn_dT=1.2e-05,
                nu_poisson_ratio=0.16,
                alpha_volume_absorption=0.001,
                intensity_reflectivity=1e-04,
                intensity_transmittance=0.999899,
                temperature=np.nan,
            ),
        ),
    ],
    standing_wave=True,
    lambda_0_laser=1.064e-06,
    t_is_trivial=True,
    p_is_trivial=True,
    use_paraxial_ray_tracing=False,
)

# %% Overview plot: geometry, mode envelope, NA and waist annotations.
plot_mirror_lens_mirror_cavity_analysis(cavity)
plt.xlim([-6e-3, 305e-3])
plt.show()

# %% Tolerance per element and parameter: the shift at which the mode overlap hits the threshold.
perturbable_params_names = ["x", "y", "phi"]
tolerance_df = cavity.generate_tolerance_dataframe(
    perturbable_params_names=perturbable_params_names
)
print(tolerance_df)
tolerance_matrix = tolerance_df.to_numpy()

# %% Overlap-vs-shift curves, scanned out to twice the tolerance, and their summary plot.
overlaps_series = cavity.generate_overlap_series(
    shifts=2 * np.abs(tolerance_matrix),
    shift_numel=30,
    perturbable_params_names=perturbable_params_names,
)
cavity.generate_overlaps_graphs(
    arm_index_for_NA=0,
    tolerance_dataframe=tolerance_df,
    overlaps_series=overlaps_series,
    perturbable_params_names=perturbable_params_names,
)
plt.show()

# %% Self-checks (keep at the end — they make this example a smoke test).
# The search may legitimately return NaN for a degenerate parameter (here: the small mirror's y),
# so we require most — not all — entries to converge.
finite = np.isfinite(tolerance_matrix)
assert (
    finite.mean() > 0.8
), f"tolerance search failed for too many parameters:\n{tolerance_df}"
assert (
    np.abs(tolerance_matrix[finite]) < 5e-2
).all(), "tolerances are suspiciously large (> 5 cm / 50 mrad)"
assert np.nanmax(overlaps_series) > 0.99, "overlap should approach 1 at zero shift"
