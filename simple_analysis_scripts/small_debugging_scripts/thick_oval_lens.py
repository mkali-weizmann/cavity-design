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
    return inside, outgoing, dict(
        back_in_air=back_in_air,
        back_in_glass=back_in_glass,
        front_in_glass=front_in_glass,
        front_in_air=front_in_air,
    )


def focus_error(outgoing, image_point):
    delta = image_point - outgoing.origin
    along = np.sum(delta * outgoing.k_vector, axis=-1)
    return np.linalg.norm(delta - along[:, np.newaxis] * outgoing.k_vector, axis=-1)


incoming, fan_angles = fan_from_the_object_point()
image_point = (T_C + FRONT_FOCAL_LENGTH) * RIGHT

print(f"Two-oval thick lens:  BFL {BACK_FOCAL_LENGTH*1e3:g} mm, FFL {FRONT_FOCAL_LENGTH*1e3:g} mm, "
      f"T_c {T_C*1e3:g} mm, n {N_GLASS}, clear aperture {DIAMETER*1e3:g} mm")
print(f"Input NA {np.sin(np.max(fan_angles)):.3f};  critical angle in the glass {np.degrees(CRITICAL_ANGLE):.1f} deg\n")
print(f"{'split':22s} {'a [mm]':>9s} {'R_back':>8s} {'R_front':>8s} "
      f"{'max i_back':>11s} {'max i_front':>12s} {'worst':>8s} {'TIR margin':>11s} {'focus err':>11s}")

results = {}
for split in SPLIT_COLORS:
    lens = build_lens(split=split)
    back, front = lens.surfaces
    _, outgoing, angles = trace(lens, incoming)
    results[split] = (lens, angles)

    worst = max(np.nanmax(angles["back_in_air"]), np.nanmax(angles["front_in_air"]))
    tir_margin = CRITICAL_ANGLE - np.nanmax(angles["front_in_glass"])
    print(f"{split:22s} {back.E_2*1e3:+9.3f} {back.radius*1e3:8.3f} {front.radius*1e3:8.3f} "
          f"{np.degrees(np.nanmax(angles['back_in_air'])):10.2f}d {np.degrees(np.nanmax(angles['front_in_air'])):11.2f}d "
          f"{np.degrees(worst):7.2f}d {np.degrees(tir_margin):10.2f}d "
          f"{focus_error(outgoing, image_point).max():11.2e}")

print("\nThe angles are quoted on the AIR side of each face, so the two columns are directly comparable.")
print("Only 'equal_deviation' brings them together - and that is exactly what makes its worst angle the")
print("smallest of the three. All three focus identically well, because both faces are exact ovals either way.")

# %% ------------------------------------------------------------------ the figure
fig = plt.figure(figsize=(13, 9))
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.32, wspace=0.24)
ax_rays = fig.add_subplot(grid[0, :])
ax_angles = fig.add_subplot(grid[1, 0])
ax_scan = fig.add_subplot(grid[1, 1])

# --- top: the ray diagram of the default lens
lens = results["equal_deviation"][0]
back, front = lens.surfaces
inside, outgoing, _ = trace(lens, incoming)
back.plot(ax=ax_rays, color="tab:blue")
front.plot(ax=ax_rays, color="tab:blue")
for i in range(N_RAYS):
    path = np.stack([incoming.origin[i], inside.origin[i], outgoing.origin[i], image_point])
    ax_rays.plot(path[:, 0], path[:, 1], color="tab:orange", lw=0.8)
ax_rays.plot(-BACK_FOCAL_LENGTH, 0, "ko", ms=7, label="object point")
ax_rays.plot(image_point[0], 0, "k*", ms=14, label="image point")
ax_rays.plot(back.focus_2[0], 0, "kx", ms=8, label=f"intermediate image (a = {back.E_2*1e3:.2f} mm, virtual)")
ax_rays.axhline(0, color="grey", lw=0.5, ls="--", zorder=0)
ax_rays.set_ylim(-2.6e-3, 2.6e-3)
ax_rays.set_title("The lens, with the 'equal_deviation' split (y stretched)", fontsize=11)
ax_rays.set_xlabel("x [mm]")
ax_rays.set_ylabel("y [mm]")
ax_rays.legend(loc="upper left", fontsize=8)

# --- bottom left: the angle at each interface, for each rule
for split, color in SPLIT_COLORS.items():
    angles = results[split][1]
    ax_angles.plot(np.degrees(fan_angles), np.degrees(angles["back_in_air"]), color=color, lw=1.6,
                   label=f"{split} - back face")
    ax_angles.plot(np.degrees(fan_angles), np.degrees(angles["front_in_air"]), color=color, lw=1.6, ls="--",
                   label=f"{split} - front face")
ax_angles.set_title("Angle of incidence at each face (air side)\nsolid = back, dashed = front", fontsize=11)
ax_angles.set_xlabel("ray angle leaving the object point [deg]")
ax_angles.set_ylabel("angle of incidence [deg]")
ax_angles.legend(fontsize=7, ncol=1, loc="upper center")
ax_angles.grid(alpha=0.3)

# --- bottom right: the worst angle over a scan of the split, with the three rules marked
scanned_a, scanned_worst = [], []
for candidate in np.linspace(0.55, 2.4, 90) * cartesian_oval_lens_intermediate_image_distance(
    BACK_FOCAL_LENGTH, FRONT_FOCAL_LENGTH, T_C, split="thin"
):
    try:
        scan_lens = build_lens(intermediate_image_distance=candidate)
        _, _, angles = trace(scan_lens, incoming)
        if not np.all(np.isfinite(angles["front_in_air"])):
            continue  # some ray is beyond the critical angle - the lens does not transmit the whole fan
        scanned_a.append(candidate)
        scanned_worst.append(max(np.nanmax(angles["back_in_air"]), np.nanmax(angles["front_in_air"])))
    except ValueError:
        continue  # the aperture outruns the oval for this split
ax_scan.plot(np.array(scanned_a) * 1e3, np.degrees(scanned_worst), color="k", lw=1.4)
for split, color in SPLIT_COLORS.items():
    angles = results[split][1]
    worst = max(np.nanmax(angles["back_in_air"]), np.nanmax(angles["front_in_air"]))
    ax_scan.plot(results[split][0].surfaces[0].E_2 * 1e3, np.degrees(worst), "o", color=color, ms=9, label=split)
ax_scan.set_title("Worst angle of incidence vs. where the\nintermediate image is put", fontsize=11)
ax_scan.set_xlabel("intermediate image distance a [mm]")
ax_scan.set_ylabel("max angle of incidence [deg]")
ax_scan.legend(fontsize=8)
ax_scan.grid(alpha=0.3)

for axis in (ax_rays,):
    axis.xaxis.set_major_formatter(lambda value, _: f"{value * 1e3:g}")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value * 1e3:g}")

fig.suptitle("A thick Cartesian-oval lens images perfectly for any power split - the split only sets the angles",
             fontsize=13)
plt.show()

# %% ------------------------------------------------------------------ self-checks
for split in SPLIT_COLORS:
    lens, angles = results[split]
    _, outgoing, _ = trace(lens, incoming)
    assert focus_error(outgoing, image_point).max() < 1e-12, split  # every split images perfectly

balanced = results["equal_deviation"][1]
worst_of = lambda angles: max(np.nanmax(angles["back_in_air"]), np.nanmax(angles["front_in_air"]))  # noqa: E731
assert worst_of(balanced) < worst_of(results["thin"][1])
assert worst_of(balanced) < worst_of(results["equal_curvature_step"][1])
print("\nAll three splits image perfectly; 'equal_deviation' has the smallest worst-case angle.")
