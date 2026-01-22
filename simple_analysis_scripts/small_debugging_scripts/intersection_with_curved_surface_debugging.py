from cavity import *
# %% # Intersection debug cell:
general_mirror = CurvedMirror(radius=5e-3, origin=np.array([0, 0, 0]), outwards_normal=np.array([-1, 0, 0]))
right_mirror = CurvedMirror(radius=5e-3, origin=np.array([0, 0, 0]), outwards_normal=np.array([1, 0, 0]))
tilt_angles = np.linspace(0, 0.15, 1000)
initial_arc_lengths = tilt_angles * general_mirror.radius
initial_rays_origin = general_mirror.parameterization(np.zeros_like(initial_arc_lengths), -initial_arc_lengths)
initial_rays_origin_alternative = np.stack((-np.cos(tilt_angles),
                                                    -np.sin(tilt_angles),
                                                    np.zeros_like(tilt_angles)), axis=1) * general_mirror.radius
orthonormal_direction = unit_vector_of_angles(theta=np.zeros_like(tilt_angles), phi=tilt_angles)
orthonormal_ray = Ray(origin=initial_rays_origin, k_vector=orthonormal_direction)
orthonormal_ray_alternative = Ray(origin=initial_rays_origin_alternative, k_vector=orthonormal_direction)
intersection_points = right_mirror.find_intersection_with_ray_exact(orthonormal_ray)
intersection_points_alternative = np.stack((np.cos(tilt_angles),
                                                   np.sin(tilt_angles),
                                                   np.zeros_like(tilt_angles)), axis=1) * general_mirror.radius