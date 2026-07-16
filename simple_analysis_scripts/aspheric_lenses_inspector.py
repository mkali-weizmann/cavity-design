# %%
from matplotlib import use
use("TkAgg")

from cavity_design import *

window = OpticalSystem(
    elements=[
    FlatRefractiveSurface(name='window_left', outwards_normal=np.array([-1.0, 0.0, 0.0]),
                          center=np.array([-1.81e-3, 0, 0]), n_1=1, n_2=1.5135, diameter=0.0082,
                          thermal_properties=MaterialProperties(temperature=np.nan)),
    FlatRefractiveSurface(name='window_right', outwards_normal=np.array([-1.0, 0.0, 0.0]),
                          center=np.array([-1.56e-3, 0.0, 0.0]), n_1=1.5135, n_2=1, diameter=0.0082,
                          thermal_properties=MaterialProperties(temperature=np.nan)),
    ],
    use_paraxial_ray_tracing=False,
    p_is_trivial=True,
    t_is_trivial=True,
    name='window'
)

EDMUND_4MM_ASPHERIC_16701 = OpticalSystem(
    elements=[
    FlatRefractiveSurface(
            center=ORIGIN,
            outwards_normal=LEFT,
            n_1=1,
            n_2=1.574, # design n: 1.576
            name="Edmund 4mm 16701 aspheric - flat side",
            diameter=6.3e-3,
        ),
        AsphericRefractiveSurface(
            center=3.65e-3 * RIGHT,
            outwards_normal=RIGHT,
            # Note: the coefficients must not be negated (coef[1] must be positive) — the curvature direction is
            # encoded in the outwards normal (LEFT here), as the AsphericSurface assertion requires.
            polynomial_coefficients=np.array([0.00000000e+00, 1.92957141e+02, 3.60108466e+06, 1.09405275e+11,
       3.76050551e+15, 4.60513590e+17, 4.99202895e+21, 5.66911316e+25,
       6.65751075e+29, 8.01870085e+33, 9.85136292e+37]),
            n_1=1.574, # design n: 1.576
            n_2=1,
            name="Edmund 4mm 16701 aspheric - convex side",
            diameter=6.3e-03,
            curvature_sign=CurvatureSigns.concave,
        ),
    ],
    use_paraxial_ray_tracing=False,
    name='Edmund 4mm Aspheric - 16701',
)

lens = EDMUND_4MM_ASPHERIC_16701
lens_inverted = EDMUND_4MM_ASPHERIC_16701.inverse
optical_system_combined = OpticalSystem(elements=[window, lens], use_paraxial_ray_tracing=False, name='optical_system')
optical_system_combined_inverted = optical_system_combined.inverse

back_focal_length = back_focal_length_of_lens_object(lens_object=lens)
plot_system = lens
plot_system_inverted = lens_inverted


# %% From collimated to point:
initial_ys = np.linspace(0, 0.0023, 10)
initial_xs = np.ones_like(initial_ys) * (0.01)
initial_positions = np.stack([initial_xs, initial_ys, np.zeros_like(initial_ys)], axis=-1)
ray_0 = Ray(origin=initial_positions, k_vector=LEFT, n=1)
rays_propagated = plot_system_inverted.propagate_ray(ray_0, propagate_with_first_surface_first=True)
plt.close('all')
ax = plot_system_inverted.plot()
# ax.scatter([plot_system[1].center[0] - back_focal_length], [0], color='red', label='Back Focal Point')
rays_propagated.plot(ax=ax)
ax.set_xlim(-0.005, 0.01)
# figure_dir = get_obsidian_save_path(filename="extract_geometry_from_zmax_params.svg")
# plt.savefig(figure_dir)
plt.show()

# %% From point to collimated
phi = np.linspace(0, 0.3, 10)
ray_origin = back_focal_length*LEFT  # optical_axis * defocus
rays_0 = Ray(origin=ray_origin, k_vector=unit_vector_of_angles(theta=0, phi=phi), n=1)
rays_propagated = plot_system.propagate_ray(rays_0, propagate_with_first_surface_first=True)
ax = plot_system.plot()
plt.xlim(-0.005, 1)
rays_propagated.plot(ax=ax)
plt.show()
