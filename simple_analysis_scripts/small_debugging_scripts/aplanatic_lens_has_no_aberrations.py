from matplotlib import use
use('TkAgg')
from cavity import *

n = 1.8
R_1 = -5e-3
R_2 = -5e-3
T_c = (1+1/n) * np.abs(R_2) - np.abs(R_1)
diameter = 7.75e-3
f = focal_length_of_lens(R_1=R_1, R_2=R_2, n=n, T_c=T_c)
back_focal_length = back_focal_length_of_lens(R_1=R_1, R_2=R_2, n=n, T_c=T_c)
first_surface_center = RIGHT * np.abs(R_1)

surface_1 = CurvedRefractiveSurface(radius=np.abs(R_1), outwards_normal=RIGHT,
                                    center=first_surface_center, n_1=1.0, n_2=n,
                                    curvature_sign=CurvatureSigns.concave, name="concave side", diameter=diameter)
surface_2 = CurvedRefractiveSurface(radius=np.abs(R_2), outwards_normal=RIGHT,
                                    center=first_surface_center + RIGHT * T_c, n_1=n, n_2=1.0,
                                    curvature_sign=CurvatureSigns.concave, name="convex side", diameter=diameter)

phi_max = 0.1
d_phi = 0.005
phi = np.arange(0, phi_max, d_phi)
ray_origin = ORIGIN
k_vectors = unit_vector_of_angles(theta=0, phi=phi)
rays_0 = Ray(origin=ray_origin, k_vector=k_vectors, n=1)
k_vectors = unit_vector_of_angles(theta=0, phi=phi)
rays_1 = surface_1.propagate_ray(ray=rays_0)
rays_1.n = n
rays_2 = surface_2.propagate_ray(ray=rays_1)
rays_2.n = 1.0

rays_history = [rays_0, rays_1, rays_2]
ray_sequence = RaySequence(rays_history)
d_0 = ray_sequence.cumulative_optical_path_length[1, 0]  # Assumes the first ray is the optical axis ray.
wavefront_points_initial = ray_sequence.parameterization(d_0, optical_path_length=True)
print(f"surface_1 center/optical axis output lens ray position: (should be the same for non-tilted case):\n{np.stack((surface_2.center, wavefront_points_initial[0, :]), axis=0)}")
R = image_of_a_point_with_thick_lens(distance_to_face_1=np.abs(R_1), R_1=R_1,
                                         R_2=R_2, n=n,
                                         T_c=T_c)  # Assumes cylindrical symmetry.
center_of_curvature = np.array(surface_2.center + LEFT * np.abs(R))
residual_distances_initial = np.abs(R) - np.linalg.norm(wavefront_points_initial - center_of_curvature, axis=-1)
polynomial_residuals_initial = Polynomial.fit(wavefront_points_initial[:, 1] ** 2, residual_distances_initial, 4).convert()

# %%
fig, ax = plt.subplots(2, 1, figsize=(18, 16))
surface_1.plot(ax=ax[0], color='blue')
surface_2.plot(ax=ax[0], color='red')
rays_0.plot(ax=ax[0], color='cyan', label='Input Rays')
rays_1.plot(ax=ax[0], color='green', label='After Surface 1')
rays_2.plot(ax=ax[0], color='magenta', label='After Surface 2')
ax[0].set_xlim(-1e-2, surface_2.center[0] + 2e-2)
ax[0].set_ylim(-diameter, diameter)
ax[0].plot(wavefront_points_initial[:, 0],
        wavefront_points_initial[:, 1],
        'kx',
        label='Wavefront Points (Initial)')
ax[0].grid()
ax[1].set_title('Residual Distances from Spherical Wavefront')
ax[1].plot(wavefront_points_initial[:, 1], residual_distances_initial)
plt.show()
