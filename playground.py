from cavity import *

# %%
cos_phi = -np.sqrt(1)
cos_theta = np.sqrt(0.97)

s = AsphericSurface(center=np.array([0.5, 0, 0]),
                    outwards_normal = np.array([cos_phi, np.sqrt(1 - cos_phi**2), 0]),
                    diameter=0.4,
                    polynomial_coefficients=[0, 1])
ray = Ray(origin=np.array([0, 0, 0]),
          k_vector=np.array([cos_theta, np.sqrt(1 - cos_theta**2), 0]))



intersection = s.find_intersection_with_ray_exact(ray)
# Validate intersection
print(s.defining_equation(intersection))

fig, ax = plt.subplots()

s.plot(ax=ax)
ray.plot(ax=ax, label='Initial Ray')
plt.plot(intersection[0], intersection[1], 'ro', label='Intersection')
plt.axis('equal')
plt.legend()
plt.show()



