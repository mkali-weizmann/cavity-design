from cavity import *

theta = 0.1
initial_ray = Ray(origin=np.array([0.0, 0.0, -100.0]), k_vector=np.array([np.cos(theta), np.sin(theta), 0]))
n=1.1
beta = np.arctan(np.sin(theta) / (n - np.cos(theta)))
single_surface = FlatRefractiveSurface(outwards_normal=np.array([np.cos(beta), -np.sin(beta), 0.0]),
                                       n_1=1,
                                       n_2=n,
                                       center=np.array([1, 0, 0]),
                                       diameter=0.4)

new_ray = single_surface.reflect_ray(initial_ray, paraxial=False)

fig, ax = plt.subplots()
initial_ray.plot(ax=ax, label='Initial Ray')
new_ray.plot(ax=ax, label='Refracted Ray')
single_surface.plot(ax=ax)
plt.axis('equal')
plt.legend()
plt.show()
print(new_ray.k_vector)

# %%
from cavity import *

theta = 0.2
initial_ray = Ray(origin=np.array([0.0, 0.0, -100.0]), k_vector=np.array([np.cos(theta), np.sin(theta), 0]))
n=1.6
beta = np.arctan(n * np.sin(theta) / (n * np.cos(theta) - 1))
single_surface = FlatRefractiveSurface(outwards_normal=np.array([np.cos(beta), np.sin(beta), 0.0]),
                                       n_1=n,
                                       n_2=1,
                                       center=np.array([1, 0, 0]),
                                       diameter=0.7)

new_ray = single_surface.reflect_ray(initial_ray, paraxial=False)

fig, ax = plt.subplots()
initial_ray.plot(ax=ax, label='Initial Ray')
new_ray.plot(ax=ax, label='Refracted Ray')
single_surface.plot(ax=ax)
plt.axis('equal')
plt.legend()
plt.show()
print(new_ray.k_vector)