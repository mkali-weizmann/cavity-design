# from matplotlib import use
# use('TkAgg')  # or 'Qt5Agg', 'Agg', etc. depending on your environment
from cavity import *
import cavity
import importlib
importlib.reload(cavity)

# globals().update({
#     name: obj
#     for name, obj in cavity.__dict__.items()
#     if not name.startswith("_")
# })

phi=0
theta=0.1

f = 20.0
T_c = 3.0
n_1 = 1
n_2 = 1.5
polynomial_coefficients = [0, 4.54546675e-02, -2.23050041e-05,
                           1.88752450e-08]  # generated for f=20, Tc=3 in aspheric_lens_generator.py
polynomial = Polynomial(polynomial_coefficients)

optical_axis = np.array([np.cos(phi), np.sin(phi), 0])

diameter = 15
back_center = f * optical_axis
front_center = back_center + T_c * optical_axis
fig, ax = plt.subplots(figsize=(15, 15))
s_1 = FlatRefractiveSurface(outwards_normal=optical_axis, center=back_center, n_1=n_1, n_2=n_2, diameter=diameter)

s_2 = AsphericRefractiveSurface(center=front_center,
                                outwards_normal=optical_axis,
                                diameter=diameter,
                                polynomial_coefficients=polynomial,
                                n_1=n_2,
                                n_2=n_1)

ray_initial = Ray(origin=np.array([[0, 0, 0], [0, 0, 0]]),
                  k_vector=np.array([[np.cos(-theta + phi), np.sin(-theta + phi), 0],
                                     # [np.cos(phi), np.sin(phi), 0],
                                     [np.cos(theta/2 + phi), np.sin(theta/2 + phi), 0]]))

intersection_1 = s_1.find_intersection_with_ray_exact(ray_initial)
ray_inner = s_1.interact_with_ray(ray_initial)

intersection_2 = s_2.find_intersection_with_ray_exact(ray_inner)
output_direction = s_2.scatter_direction_exact(ray_inner)
ray_output = s_2.interact_with_ray(ray_inner)

s_1.plot(ax=ax, label='Back Surface', color='black')
ray_initial.plot(ax=ax, label='Initial Ray', color='m')

ray_inner.plot(ax=ax, label='Inner Ray', color='c')
ax.legend()
plt.axis('equal')
ax.grid()
ax.set_title(f"{ray_output.k_vector @ optical_axis}\n{ray_initial.k_vector @ optical_axis}")
plt.show()

# %%
# fig, ax = plt.subplots()
# s.plot(ax=ax)
# ray.plot(ax=ax, label='Initial Ray')
# plt.plot(intersection[0], intersection[1], 'ro', label='Intersection')
# plt.plot([intersection[0], intersection[0] + n[0]*1], [intersection[1], intersection[1] + n[1]*1], 'g-', label='Normal Vector')
# plt.axis('equal')
# plt.legend()
# plt.show()



