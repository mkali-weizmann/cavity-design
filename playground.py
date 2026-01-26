from cavity import *

# Parameters
phi = np.linspace(0, 0.1, 10)
k_vectors = unit_vector_of_angles(theta=0, phi=phi)
# %%
ray_origin = np.array([0.0, 0.0, 0.0])
optical_axis = np.array([1, 0, 0])
f = 5e-3
T_c = 3.0e-3
n = 1.5
diameter = 7.75e-3
back_center = ray_origin + f * optical_axis

# Objects:
rays = Ray(origin=ray_origin, k_vector=k_vectors)
flat_surface, aspheric_surface = Surface.from_params(generate_aspheric_lens_params(f=f,
                                                                              T_c=T_c,
                                                                              n=n,
                                                                              forward_normal=optical_axis,
                                                                              diameter=diameter,
                                                                              polynomial_degree=8,
                                                                              flat_faces_center=back_center,
                                                                              name="aspheric_lens_automatic"))
# Trace rays through the lens:
rays_after_flat_surface = flat_surface.propagate_ray(rays)
rays_after_aspheric_surface = aspheric_surface.propagate_ray(rays_after_flat_surface)
# %%
fig, ax = plt.subplots()
rays.plot(ax=ax, label='Before lens', color='black')
rays_after_flat_surface.plot(ax=ax, label='After flat surface', color='blue')
rays_after_aspheric_surface.plot(ax=ax, label='After aspheric surface', color='red')
flat_surface.plot(ax=ax, color='green')
aspheric_surface.plot(ax=ax, color='orange')
ax.set_xlim(-1e-3, 50e-3)
plt.show()
