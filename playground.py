from cavity import *

# Parameters
phi = np.linspace(0, 0.2, 10)
k_vectors = unit_vector_of_angles(theta=0, phi=phi)
# %%
ray_origin = np.array([0.0, 0.0, 0.0])
optical_axis = np.array([1, 0, 0])
f = 5e-3
defocus = 0.1e-3
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

# %%


# %%
ray_sequence = RaySequence(rays_history)
points = ray_sequence.parameterization(12e-3, optical_path_length=True)
# plot points:
fig, ax = plt.subplots()
rays_0.plot(ax=ax, label='Before lens', color='black', linewidth=0.5)
rays_1.plot(ax=ax, label='After flat surface', color='blue', linewidth=0.5)
rays_2.plot(ax=ax, label='After aspheric surface', color='red', linewidth=0.5)
flat_surface.plot(ax=ax, color='green')
aspheric_surface.plot(ax=ax, color='orange')
ax.set_xlim(-1e-3, 50e-3)
plt.scatter(points[:, 0], points[:, 1], s=8, color='purple')
plt.show()



