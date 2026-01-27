from matplotlib import use
use('TkAgg')  # or 'Qt5Agg', 'GTK3Agg', etc. depending on your system
from cavity import *

# Parameters
phi = np.linspace(0, 0.2, 10)
k_vectors = unit_vector_of_angles(theta=0, phi=phi)
# %%
ray_origin = np.array([0.0, 0.0, 0.0])
optical_axis = np.array([1, 0, 0])
f = 5e-3
defocus = 3e-4
T_c = 3.0e-3
n = 1.5
diameter = 7.75e-3
back_center = ray_origin + (f+defocus) * optical_axis

# Objects:
rays_0 = Ray(origin=ray_origin, k_vector=k_vectors)
flat_surface, aspheric_surface = Surface.from_params(generate_aspheric_lens_params(f=f,
                                                                              T_c=T_c,
                                                                              n=n,
                                                                              forward_normal=optical_axis,
                                                                              diameter=diameter,
                                                                              polynomial_degree=8,
                                                                              flat_faces_center=back_center,
                                                                              name="aspheric_lens_automatic"))
# Trace rays through the lens:
rays_1 = flat_surface.propagate_ray(rays_0)
rays_2 = aspheric_surface.propagate_ray(rays_1)
rays_0.n = 1
rays_1.n = n
rays_2.n = 1
rays_history = [rays_0, rays_1, rays_2]

ray_sequence = RaySequence(rays_history)
d_0 = ray_sequence.cumulative_optical_path_length[1, 0]
points = ray_sequence.parameterization(d_0, optical_path_length=True)

# plot points:
fig, ax = plt.subplots()
rays_0.plot(ax=ax, label='Before lens', color='black', linewidth=0.5)
rays_1.plot(ax=ax, label='After flat surface', color='blue', linewidth=0.5)
rays_2.plot(ax=ax, label='After aspheric surface', color='red', linewidth=0.5)
flat_surface.plot(ax=ax, color='green')
aspheric_surface.plot(ax=ax, color='orange')
ax.set_xlim(-1e-3, 100e-3)
ax.set_ylim(-4.2e-3, 4.2e-3)
ax.grid()
plt.scatter(points[:, 0], points[:, 1], s=8, color='purple')
plt.show()


R, center = extract_matching_sphere(points[..., 0, :], points[..., 1, :], rays_0.k_vector[..., 0, :])
points_rel = points - center
phi_dummy = np.linspace(0, np.arctan(points_rel[-1,1] / points_rel[-1,0]), 100)
sphere_dummy_points = center - R * np.stack((np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)), axis=-1)
points_relative_distances = np.linalg.norm(points - center, axis=-1) - R
fig, ax = plt.subplots(2, 1)
ax[0].plot(points[:, 1] * 1e3, points_relative_distances * 1e6)
ax[0].set_xlabel('y (mm)')
ax[0].set_ylabel('phase front relative distance to sphere (µm)')
ax[0].grid()
ax[1].plot(points[:, 0] * 1e3, points[:, 1] * 1e3)
ax[1].plot(sphere_dummy_points[:, 0] * 1e3, sphere_dummy_points[:, 1] * 1e3, linestyle='dashed')
ax[1].set_xlabel('x (mm)')
ax[1].set_ylabel('y (mm)')
ax[1].grid()
plt.show()

# %%
