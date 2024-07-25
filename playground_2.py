from cavity import *

convex_right = CurvedSurface(radius=2, outwards_normal=np.array([1, 0, 0]), center=np.array([1, 0, 0]), curvature_sign=CurvatureSigns.concave)
concave_right = CurvedSurface(radius=2, outwards_normal=np.array([-1, 0, 0]), center=np.array([1, 0, 0]), curvature_sign=CurvatureSigns.convex)
convex_left = CurvedSurface(radius=2, outwards_normal=np.array([-1, 0, 0]), center=np.array([-1, 0, 0]), curvature_sign=CurvatureSigns.concave)
concave_left = CurvedSurface(radius=2, outwards_normal=np.array([1, 0, 0]), center=np.array([-1, 0, 0]), curvature_sign=CurvatureSigns.convex)

ray_right = Ray(origin=np.array([0, 0, 0]), k_vector=np.array([1, 0, 0]))
ray_left = Ray(origin=np.array([0, 0, 0]), k_vector=np.array([-1, 0, 0]))

ray = ray_left
surface = concave_left

a = surface.find_intersection_with_ray_exact(ray)
fig, ax = plt.subplots()
ray.plot(ax, label='ray')
surface.plot(ax, label='surface')
ax.scatter(a[0], a[1], label='Intersection')
ax.scatter(ray.origin[0], ray.origin[1], label='origin')
ax.quiver(ray.origin[0], ray.origin[1], ray.k_vector[0], ray.k_vector[1], label='k_vector')
plt.legend()
plt.show()

# %%
concave_right.find_intersection_with_ray_exact(ray_right)
convex_right.find_intersection_with_ray_exact(ray_left)
concave_right.find_intersection_with_ray_exact(ray_left)
convex_left.find_intersection_with_ray_exact(ray_left)
concave_left.find_intersection_with_ray_exact(ray_left)
convex_left.find_intersection_with_ray_exact(ray_right)
concave_left.find_intersection_with_ray_exact(ray_left)


