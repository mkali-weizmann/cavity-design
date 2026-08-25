# %%
"""Build the manufacturable version of an oval singlet, as a lens you can drop into a simulation.

The pipeline, which is the point of the script:

    Cartesian oval  ->  vendor even asphere (R, k, alpha_4, alpha_6)  ->  polynomial of degree 20
                    ->  AsphericRefractiveSurface x2  ->  `manufactured_lens`

The middle step is the lossy one and the one worth simulating: it is what a polishing shop is actually
given. The last step is bookkeeping - this library's aspheres are pure polynomials in rho**2, so the conic
base has to be re-expressed before it can be traced, and that re-expression should cost far less than the
truncation above it.

`manufactured_lens` is left FLOATING - place it yourself with .to_position(...), like a catalog element.
`exact_lens` is the ideal oval it came from, for comparison.
"""

from cavity_design import *

BACK_FOCAL_LENGTH = 5e-3  # Object distance, from the back (first) vertex.
FRONT_FOCAL_LENGTH = 200e-3  # Image distance, from the front (second) vertex.
CENTER_THICKNESS = 3e-3
REFRACTIVE_INDEX = 1.5
DIAMETER = 7.75e-3
N_ALPHA = 2  # Correction terms the vendor is given: alpha_4, alpha_6.
DEGREE = 20  # Degree of the polynomial realization. Simulation only, so manufacturability is not a concern.

# method="fit" rather than the default "taylor": this lens is steep enough that its conic base has no
# convergent Taylor series over the aperture (|1+k|*(rho_max/R)**2 = 2.3 on the back face), so expanding
# about the vertex diverges however high the degree goes. A least-squares fit over the aperture has no such
# limit, and is the better choice for simulation anyway - it spreads its error rather than piling it at the
# rim. Either way the vertex curvature is pinned, so the paraxial power is preserved exactly.
EXPANSION_METHOD = "fit"

exact_lens = generate_cartesian_oval_lens(
    back_focal_length=BACK_FOCAL_LENGTH,
    front_focal_length=FRONT_FOCAL_LENGTH,
    T_c=CENTER_THICKNESS,
    n=REFRACTIVE_INDEX,
    diameter=DIAMETER,
    name="oval singlet (exact)",
)

specs = []
faces = []
for surface in exact_lens.to_position(ORIGIN).surfaces:
    spec = surface.vendor_aspheric_parameters(n_alpha=N_ALPHA)
    specs.append(spec)
    faces.append(
        AsphericRefractiveSurface(
            name=f"{surface.name} (as manufactured)",
            # Floating, with the second face at its relative offset - the convention catalog elements use, so
            # that .to_position() places the whole lens by its first vertex.
            center=None if not faces else CENTER_THICKNESS * RIGHT * 1j,
            outwards_normal=surface.outwards_normal,
            polynomial_coefficients=spec.to_polynomial_coefficients(degree=DEGREE, method=EXPANSION_METHOD),
            curvature_sign=surface.curvature_sign,
            n_1=surface.n_1,
            n_2=surface.n_2,
            diameter=surface.diameter,
            material_properties=surface.material_properties,
        )
    )

manufactured_lens = OpticalSystem(elements=faces, use_paraxial_ray_tracing=False, t_is_trivial=True, p_is_trivial=True)

# %%
print(
    f"Oval singlet: {BACK_FOCAL_LENGTH * 1e3:g} mm -> {FRONT_FOCAL_LENGTH * 1e3:g} mm, n = {REFRACTIVE_INDEX}, "
    f"T_c = {CENTER_THICKNESS * 1e3:g} mm, clear aperture {DIAMETER * 1e3:g} mm"
)
print(f"Vendor spec: {N_ALPHA} correction terms. Realized as a degree-{DEGREE} polynomial ({EXPANSION_METHOD}).\n")

for name, spec in zip(("BACK (first) face", "FRONT (second) face"), specs):
    print("=" * 78)
    print(name)
    print(spec)
    print()

# How much of the ideal surface survived each step, so the simulation is read with the right expectations.
# The first number is the real, physical cost of truncating the vendor spec - it is what the manufactured
# part will genuinely differ by. The second is only the polynomial bookkeeping, and should be negligible
# beside it; if it is not, raise DEGREE.
print("=" * 78)
rho = np.linspace(0, DIAMETER / 2, 401)
for name, oval, spec, face in zip(("back", "front"), exact_lens.to_position(ORIGIN).surfaces, specs, faces):
    exact_sag = oval.curvature_sign * oval.local_sag(rho)
    vendor_sag = spec.sag(rho)
    realized_sag = oval.curvature_sign * face.polynomial(rho**2)
    print(
        f"{name:5s} face:  oval -> vendor spec {np.max(np.abs(vendor_sag - exact_sag)) * 1e9:9.4f} nm"
        f"   |   vendor spec -> polynomial {np.max(np.abs(realized_sag - vendor_sag)) * 1e12:9.4f} pm"
    )
    assert np.isclose(face.radius, oval.radius, rtol=1e-12), "vertex radius must survive the round trip"

print()
print(
    f"manufactured_lens: {len(manufactured_lens.elements)} aspheric surfaces, floating "
    f"(positions_defined = {manufactured_lens.positions_defined})."
)
print("Place it before tracing, e.g.:  lens = manufactured_lens.to_position(x * RIGHT)")
