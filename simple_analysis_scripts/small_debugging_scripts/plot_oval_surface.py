# %%
"""Point source -> point image with a Cartesian oval.

A ``RefractiveCartesianOval`` is the exact surface that images one pair of conjugate points with no
spherical aberration at all: *every* ray leaving the object focus is refracted precisely through the
image focus, however far off axis it starts. A spherical or polynomial-aspheric surface only manages
this approximately.

The surface is defined by two signed focal distances, measured from the vertex along the direction the
light travels, and by the two refractive indices - which set its *shape*, not just its refraction:

  * ``E_1 > 0`` - a real object: the incoming rays diverge from ``focus_1``, behind the surface.
  * ``E_2 > 0`` - a real image: the refracted rays physically converge to ``focus_2``, in front of it.
  * ``E_2 < 0`` - a virtual image: the refracted rays diverge, and it is their backward extensions that
    meet at ``focus_2``, behind the surface.

This script runs both signs of ``E_2`` side by side and prints the residual focus error for each.
"""
from matplotlib import use
use("QT5Agg")  # for interactive plotting in Spyder
from cavity_design import *

N_RAYS = 13
HALF_ANGLE = 0.16  # rad - the half-angle of the cone leaving the point source


def fan_from_point_source(surface, half_angle=HALF_ANGLE, n_rays=N_RAYS):
    """A fan of rays leaving the object focus of ``surface``, spread symmetrically about the optical axis."""
    optical_axis = surface.propagation_direction
    transverse = np.cross(optical_axis, np.array([0.0, 0.0, 1.0]))
    angles = np.linspace(-half_angle, half_angle, n_rays)
    k_vector = np.stack([np.cos(angle) * optical_axis + np.sin(angle) * transverse for angle in angles])
    return Ray(origin=np.tile(surface.focus_1, (n_rays, 1)), k_vector=k_vector, n=surface.n_1)


def focus_error(outgoing_ray, image_point):
    """Perpendicular distance from the image point to each outgoing ray's (infinite) line.

    Measuring against the line rather than the forward half-line makes this work for a virtual image
    too, where the rays only meet the image point when extended backwards."""
    delta = image_point - outgoing_ray.origin
    along = np.sum(delta * outgoing_ray.k_vector, axis=-1)
    return np.linalg.norm(delta - along[:, np.newaxis] * outgoing_ray.k_vector, axis=-1)


def demonstrate(ax, title, E_1, E_2, diameter, n_1=1.0, n_2=1.5):
    surface = RefractiveCartesianOval(
        center=ORIGIN,
        outwards_normal=LEFT,  # convex side faces the source, so the light travels to the right
        E_1=E_1,
        E_2=E_2,
        n_1=n_1,
        n_2=n_2,
        diameter=diameter,
        name=title,
    )
    source, image = surface.focus_1, surface.focus_2
    incoming = fan_from_point_source(surface)
    outgoing = surface.propagate_ray(incoming)
    errors = focus_error(outgoing, image)

    numerical_aperture = n_1 * np.sin(HALF_ANGLE)
    largest_ray_height = surface.radial_distance_from_axis(outgoing.origin).max()
    print(f"\n{title}")
    print(f"    E_1 = {E_1 * 1e3:+.1f} mm (source), E_2 = {E_2 * 1e3:+.1f} mm (image), n: {n_1} -> {n_2}")
    print(f"    source at {np.round(source * 1e3, 3)} mm, image at {np.round(image * 1e3, 3)} mm")
    print(f"    vertex radius of curvature : {surface.radius * 1e3:.3f} mm")
    print(f"    curvature_sign             : {surface.curvature_sign:+d} (derived from the optics)")
    print(f"    input NA                   : {numerical_aperture:.3f}")
    print(f"    largest ray height         : {largest_ray_height * 1e3:.3f} mm")
    print(f"    WORST FOCUS ERROR          : {errors.max():.2e} m  <- a point, to machine precision")

    # --- draw it
    surface.plot(ax=ax, color="tab:blue")
    for i in range(N_RAYS):
        hit = outgoing.origin[i]
        ax.plot([incoming.origin[i, 0], hit[0]], [incoming.origin[i, 1], hit[1]], color="tab:orange", lw=0.9)
        if E_2 > 0:
            # Real image: the ray physically travels on to the image point.
            ax.plot([hit[0], image[0]], [hit[1], image[1]], color="tab:red", lw=0.9)
        else:
            # Virtual image: the ray diverges forwards, and its backward extension reaches the image point.
            forward = hit + abs(E_2) * outgoing.k_vector[i]
            ax.plot([hit[0], forward[0]], [hit[1], forward[1]], color="tab:red", lw=0.9)
            ax.plot([hit[0], image[0]], [hit[1], image[1]], color="tab:red", lw=0.6, ls=":", alpha=0.6)

    ax.plot(source[0], source[1], "ko", ms=7, label="point source (focus_1)")
    ax.plot(image[0], image[1], "k*", ms=15, label=f"point image (focus_2), {'real' if E_2 > 0 else 'virtual'}")
    ax.axhline(0, color="grey", lw=0.5, ls="--", zorder=0)
    ax.set_title(f"{title}      worst focus error = {errors.max():.1e} m", fontsize=11)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]  (exaggerated)")
    # The scene is ~90 mm long but only ~12 mm tall, so an equal aspect ratio would squash it into an
    # unreadable strip. Stretching y is the usual convention for a ray diagram.
    ax.set_ylim(-1.9 * diameter / 2, 1.9 * diameter / 2)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value * 1e3:g}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 1e3:g}")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    return errors.max()


fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

real_error = demonstrate(
    axes[0],
    "Real image  (E_2 > 0)",
    E_1=30e-3,
    E_2=60e-3,
    diameter=12e-3,
)
virtual_error = demonstrate(
    axes[1],
    "Virtual image  (E_2 < 0)",
    E_1=30e-3,
    E_2=-60e-3,
    diameter=12e-3,
)

fig.suptitle("A Cartesian oval maps a point source to a point image, exactly", fontsize=13)
plt.tight_layout()
plt.show()

# Both cases must be perfect to numerical precision - that is the whole point of this surface.
assert real_error < 1e-12, real_error
assert virtual_error < 1e-12, virtual_error
print("\nBoth conjugate pairs image perfectly.")
