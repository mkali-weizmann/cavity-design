# %%
"""A thick lens of two Cartesian ovals, and how the angles of incidence land on its two faces.

``generate_cartesian_oval_lens`` builds a floating two-face lens that images one conjugate pair with no
spherical aberration at all: each face is an exact Cartesian oval, the back one imaging the object onto an
intermediate point and the front one imaging that point onto the final image. Because both halves are exact,
the composite is exact *wherever the intermediate point is put* - the split of the work between the two faces
is a free parameter that costs nothing in image quality.

What it does cost is the angles of incidence, and with them the Fresnel loss at each face and the margin
before total internal reflection on the way out. This script traces one fan through the lens under all three
splitting rules and compares those angles:

  * ``"thin"``                 - a = -2/K, the curvature after the back face set to the mean of the entrance
                                 and exit curvatures. Exact as T_c -> 0.
  * ``"equal_curvature_step"`` - equal jumps in curvature, including its drift across the glass.
  * ``"equal_deviation"``      - equal ray *deviation* at the two faces, which (both faces having the same
                                 index ratio) is the same as equal angles of incidence. The default.

The derivation of all three, with the notation, is in theory/cartesian_oval_lens_power_split.md.
"""
from matplotlib import use

try:
    use("QtAgg")  # an interactive window in Spyder; QtAgg rather than QT5Agg so it takes whichever binding is
except ImportError:  # installed (PySide6 here, PyQt5 under Spyder). Skipped when there is no GUI at all.
    pass
from cavity_design import *

BACK_FOCAL_LENGTH = 5e-3  # object point, this far in front of the back face's vertex
FRONT_FOCAL_LENGTH = 50e-3  # image point, this far past the front face's vertex
T_C = 4e-3
N_GLASS = 1.5
DIAMETER = 8e-3
MARGINAL_RAY_HEIGHT = 1.2e-3  # how high up the back face the outermost ray of the fan lands
N_RAYS = 15

CRITICAL_ANGLE = np.arcsin(1 / N_GLASS)  # beyond this, the front face reflects instead of transmitting
SPLIT_COLORS = {"thin": "tab:orange", "equal_curvature_step": "tab:green", "equal_deviation": "tab:blue"}


def build_lens(split="equal_deviation", intermediate_image_distance=None):
    """The lens, placed with its back vertex at the origin. Floating until `.to_position` is called."""
    return generate_cartesian_oval_lens(
        back_focal_length=BACK_FOCAL_LENGTH,
        front_focal_length=FRONT_FOCAL_LENGTH,
        T_c=T_C,
        n=N_GLASS,
        diameter=DIAMETER,
        split=split,
        intermediate_image_distance=intermediate_image_distance,
    ).to_position(ORIGIN)


def fan_from_the_object_point(n_rays=N_RAYS):
    """A fan out of the object point, sized so its marginal ray reaches MARGINAL_RAY_HEIGHT on the back face."""
    half_angle = np.arctan(MARGINAL_RAY_HEIGHT / BACK_FOCAL_LENGTH)
    angles = np.linspace(-half_angle, half_angle, n_rays)
    k_vector = np.stack([np.cos(a) * RIGHT + np.sin(a) * UP for a in angles])
    origin = np.tile(BACK_FOCAL_LENGTH * LEFT, (n_rays, 1))
    return Ray(origin=origin, k_vector=k_vector, n=1.0), angles


def angle_to_the_normal(surface, ray, hit_point):
    """Angle between a ray and the surface normal, in whatever medium the ray is currently travelling."""
    cosine = np.abs(np.sum(surface.normal_at_a_point(hit_point) * ray.k_vector, axis=-1))
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def trace(lens, incoming):
    """Push the fan through both faces and collect the incidence angle at each, on both sides of each face."""
    back, front = lens.surfaces
    inside = back.propagate_ray(incoming)
    outgoing = front.propagate_ray(inside)

    # The back face is hit from the air, the front face from inside the glass - so one angle of each pair is
    # measured directly and the other follows from Snell's law.
    back_in_air = angle_to_the_normal(back, incoming, inside.origin)
    back_in_glass = np.arcsin(np.sin(back_in_air) / N_GLASS)
    front_in_glass = angle_to_the_normal(front, inside, outgoing.origin)
    sine_out = N_GLASS * np.sin(front_in_glass)
    # Above the critical angle there is no transmitted ray at all; nan says so rather than clipping it away.
    front_in_air = np.where(sine_out <= 1, np.arcsin(np.clip(sine_out, -1, 1)), np.nan)
    return (
        inside,
        outgoing,
        dict(
            back_in_air=back_in_air,
            back_in_glass=back_in_glass,
            front_in_glass=front_in_glass,
            front_in_air=front_in_air,
        ),
    )


def draw_face(ax, surface, color="tab:blue"):
    """The profile of one face, drawn from its own sag. Straight from local_sag rather than surface.plot(), which
    would also draw the dashed back plane of a stand-alone element - clutter when the faces belong to a lens."""
    rho = np.linspace(-surface.diameter / 2, surface.diameter / 2, 300)
    transverse = np.cross(surface.outwards_normal, np.array([0.0, 0.0, 1.0]))
    points = (
        surface.center
        + transverse * rho[:, np.newaxis]
        + surface.local_sag(np.abs(rho))[:, np.newaxis] * surface.inwards_normal
    )
    ax.plot(points[:, 0], points[:, 1], color=color, lw=1.5)


def focus_error(outgoing, image_point):
    delta = image_point - outgoing.origin
    along = np.sum(delta * outgoing.k_vector, axis=-1)
    return np.linalg.norm(delta - along[:, np.newaxis] * outgoing.k_vector, axis=-1)


def worst_angle(angles):
    """The larger of the two angles of incidence, both quoted on the air side so they are comparable."""
    return max(np.nanmax(angles["back_in_air"]), np.nanmax(angles["front_in_air"]))


incoming, fan_angles = fan_from_the_object_point()
image_point = (T_C + FRONT_FOCAL_LENGTH) * RIGHT

print(
    f"Two-oval thick lens:  BFL {BACK_FOCAL_LENGTH * 1e3:g} mm, FFL {FRONT_FOCAL_LENGTH * 1e3:g} mm, "
    f"T_c {T_C * 1e3:g} mm, n {N_GLASS}, clear aperture {DIAMETER * 1e3:g} mm"
)
print(
    f"Input NA {np.sin(np.max(fan_angles)):.3f};  "
    f"critical angle in the glass {np.degrees(CRITICAL_ANGLE):.1f} deg\n"
)
print(
    f"{'split':22s} {'a [mm]':>9s} {'R_back':>8s} {'R_front':>8s} "
    f"{'max i_back':>11s} {'max i_front':>12s} {'worst':>8s} {'TIR margin':>11s} {'focus err':>11s}"
)

results = {}
for split in SPLIT_COLORS:
    lens = build_lens(split=split)
    back, front = lens.surfaces
    _, outgoing, angles = trace(lens, incoming)
    results[split] = (lens, angles)

    tir_margin = CRITICAL_ANGLE - np.nanmax(angles["front_in_glass"])
    print(
        f"{split:22s} {back.E_2 * 1e3:+9.3f} {back.radius * 1e3:8.3f} {front.radius * 1e3:8.3f} "
        f"{np.degrees(np.nanmax(angles['back_in_air'])):10.2f}d "
        f"{np.degrees(np.nanmax(angles['front_in_air'])):11.2f}d "
        f"{np.degrees(worst_angle(angles)):7.2f}d {np.degrees(tir_margin):10.2f}d "
        f"{focus_error(outgoing, image_point).max():11.2e}"
    )

print("\nThe angles are quoted on the AIR side of each face, so the two columns are directly comparable.")
print("Only 'equal_deviation' brings them together - and that is exactly what makes its worst angle the")
print("smallest of the three. All three focus identically well, because both faces are exact ovals either way.")

# %% ------------------------------------------------------------------ the figure
fig = plt.figure(figsize=(13, 11))
grid = fig.add_gridspec(3, 2, height_ratios=[0.8, 1.2, 0.9], hspace=0.38, wspace=0.22)
ax_rays = fig.add_subplot(grid[0, :])
ax_zoom = fig.add_subplot(grid[1, 0])
ax_angles = fig.add_subplot(grid[1, 1])
ax_scan = fig.add_subplot(grid[2, :])

lens = results["equal_deviation"][0]
back, front = lens.surfaces
inside, outgoing, angles = trace(lens, incoming)

# --- top: object point -> lens -> image point, and the same again zoomed onto the lens itself
for axis in (ax_rays, ax_zoom):
    draw_face(axis, back)
    draw_face(axis, front)
    for i in range(N_RAYS):
        path = np.stack([incoming.origin[i], inside.origin[i], outgoing.origin[i], image_point])
        axis.plot(path[:, 0], path[:, 1], color="tab:orange", lw=0.8)
    axis.axhline(0, color="grey", lw=0.5, ls="--", zorder=0)
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.xaxis.set_major_formatter(lambda value, _: f"{value * 1e3:g}")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value * 1e3:g}")

ax_rays.plot(-BACK_FOCAL_LENGTH, 0, "ko", ms=7, label="object point")
ax_rays.plot(image_point[0], 0, "k*", ms=14, label="image point")
ax_rays.plot(back.focus_2[0], 0, "kx", ms=8, label=f"intermediate image (a = {back.E_2 * 1e3:.2f} mm, virtual)")
ax_rays.set_ylim(-2.6e-3, 2.6e-3)
ax_rays.set_title("The whole path, for the 'equal_deviation' split (y stretched)", fontsize=11)
ax_rays.legend(loc="upper right", fontsize=8, framealpha=0.9)

# --- middle left: the lens to scale, with the normal drawn where the marginal ray strikes each face
ax_zoom.set_xlim(-1.3e-3, T_C + 1.3e-3)
ax_zoom.set_ylim(-2.2e-3, 4.0e-3)
ax_zoom.set_aspect("equal")  # here the shapes and the angles are the point, so nothing may be stretched
marginal = N_RAYS - 1  # the outermost ray of the fan
# Each face is labelled with BOTH of its angles. Quoting the back face in air against the front face in the
# glass would compare across a refractive index and make a balanced pair look wildly lopsided; it is air
# against air, or glass against glass, that has to match.
for surface, hit_point, label, in_air, in_glass, text_x in (
    (back, inside.origin[marginal], "back", angles["back_in_air"][marginal], angles["back_in_glass"][marginal], 0.0),
    (
        front,
        outgoing.origin[marginal],
        "front",
        angles["front_in_air"][marginal],
        angles["front_in_glass"][marginal],
        4.2e-3,
    ),
):
    normal = surface.normal_at_a_point(hit_point)
    segment = np.stack([hit_point - 1.0e-3 * normal, hit_point + 1.0e-3 * normal])
    ax_zoom.plot(segment[:, 0], segment[:, 1], color="grey", lw=1.0, ls=":")
    ax_zoom.plot(*hit_point[:2], "o", color="tab:red", ms=5)
    ax_zoom.annotate(
        f"{label} face\n{np.degrees(in_air):.1f}$^\\circ$ in air\n{np.degrees(in_glass):.1f}$^\\circ$ in glass",
        xy=hit_point[:2],
        xytext=(text_x, 3.9e-3),
        fontsize=8.5,
        ha="center",
        va="top",
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.7),
    )
ax_zoom.set_title(
    "The lens to scale, with the surface normal (dotted) where the\nmarginal ray strikes. Compare air with air, "
    "glass with glass.",
    fontsize=11,
)

# --- middle right: the angle at each interface, for each rule. The fan is symmetric, so only half is drawn.
upper_half = fan_angles >= 0
for split, color in SPLIT_COLORS.items():
    split_angles = results[split][1]
    ax_angles.plot(
        np.degrees(fan_angles[upper_half]),
        np.degrees(split_angles["back_in_air"][upper_half]),
        color=color,
        lw=1.7,
        label=f"{split} - back",
    )
    ax_angles.plot(
        np.degrees(fan_angles[upper_half]),
        np.degrees(split_angles["front_in_air"][upper_half]),
        color=color,
        lw=1.7,
        ls="--",
        label=f"{split} - front",
    )
ax_angles.set_title("Angle of incidence at each face, on the air side\nsolid = back, dashed = front", fontsize=11)
ax_angles.set_xlabel("ray angle leaving the object point [deg]")
ax_angles.set_ylabel("angle of incidence [deg]")
ax_angles.legend(fontsize=7.5, loc="upper left")
ax_angles.grid(alpha=0.3)

# --- bottom: the worst angle over a scan of the split, with the three rules marked
scanned_a, scanned_worst = [], []
for candidate in np.linspace(0.55, 2.4, 120) * cartesian_oval_lens_intermediate_image_distance(
    BACK_FOCAL_LENGTH, FRONT_FOCAL_LENGTH, T_C, split="thin"
):
    try:
        _, _, scan_angles = trace(build_lens(intermediate_image_distance=candidate), incoming)
    except ValueError:
        continue  # the aperture outruns the oval for this split
    if not np.all(np.isfinite(scan_angles["front_in_air"])):
        continue  # some ray is past the critical angle - the lens does not transmit the whole fan
    scanned_a.append(candidate)
    scanned_worst.append(worst_angle(scan_angles))
ax_scan.plot(np.array(scanned_a) * 1e3, np.degrees(scanned_worst), color="k", lw=1.4)
for split, color in SPLIT_COLORS.items():
    ax_scan.plot(
        results[split][0].surfaces[0].E_2 * 1e3,
        np.degrees(worst_angle(results[split][1])),
        "o",
        color=color,
        ms=10,
        label=split,
    )
ax_scan.set_title(
    "Worst angle of incidence against where the intermediate image is put - "
    "the split is free, so this curve is the only thing it decides",
    fontsize=11,
)
ax_scan.set_xlabel("intermediate image distance a [mm]")
ax_scan.set_ylabel("max angle of incidence [deg]")
ax_scan.legend(fontsize=9)
ax_scan.grid(alpha=0.3)

fig.suptitle(
    "A thick Cartesian-oval lens images perfectly for any power split - the split only sets the angles",
    fontsize=13,
)
plt.show()

# %% ------------------------------------------------------------------ self-checks
for split in SPLIT_COLORS:
    _, checked_outgoing, _ = trace(results[split][0], incoming)
    assert focus_error(checked_outgoing, image_point).max() < 1e-12, split  # every split images perfectly

balanced = results["equal_deviation"][1]
assert np.isclose(np.nanmax(balanced["back_in_air"]), np.nanmax(balanced["front_in_air"]), rtol=0.05)
assert worst_angle(balanced) < worst_angle(results["thin"][1])
assert worst_angle(balanced) < worst_angle(results["equal_curvature_step"][1])
print("\nAll three splits image perfectly; 'equal_deviation' has the smallest worst-case angle.")
