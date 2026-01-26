from cavity import *
# Run once per console session
# %load_ext autoreload
# %autoreload 2
cos_phi = np.sqrt(1)
cos_theta = np.sqrt(0.95)

polynomial_coefficients = [0, 1]
polynomial = Polynomial(polynomial_coefficients)
n_1 = 1
n_2 = 1.5
diameter = 0.4
back_center = np.array([0.5, 0, 0])
center_thickness = polynomial(diameter**2 / 4)
front_center = back_center + np.array([center_thickness, 0, 0])
forwards_normal = np.array([cos_phi, np.sqrt(1 - cos_phi**2), 0])
fig, ax = plt.subplots()
s_1 = FlatRefractiveSurface(outwards_normal=forwards_normal, center=back_center,n_1=n_1, n_2=n_2, diameter=diameter)

s_2 = AsphericRefractiveSurface(center=front_center,
                    outwards_normal = np.array([cos_phi, np.sqrt(1 - cos_phi**2), 0]),
                    diameter=0.4,
                    polynomial_coefficients=polynomial,
                              n_1=1.5,
                              n_2=1)

ray_initial = Ray(origin=np.array([0, 0, 0]),
          k_vector=np.array([cos_theta, np.sqrt(1 - cos_theta**2), 0]))


intersection_1 = s_1.find_intersection_with_ray_exact(ray_initial)
ray_inner = s_1.propagate_ray(ray_initial)

intersection_2 = s_2.find_intersection_with_ray_exact(ray_inner)
output_direction = s_2.scatter_direction_exact(ray_inner)
ray_output = s_2.propagate_ray(ray_inner)

s_1.plot(ax=ax, label='Back Surface', color='black')
ray_initial.plot(ax=ax, label='Initial Ray', color='m')

ray_inner.plot(ax=ax, label='Inner Ray', color='c')
ax.legend()
ax.grid()
plt.show()