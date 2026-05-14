from cavity_design.analyze_potential import *

def betas_of_NA(NAs, n):
    return np.arcsin(NAs/np.sqrt(n**2+1-2*np.sqrt(n**2-NAs**2)))

def α_D_collimated(β_value, n):
    q = np.sin(β_value) * (np.sqrt(np.maximum(n ** 2 - np.sin(β_value) ** 2, 0.0)) - np.cos(β_value))
    α = np.arcsin(q)
    return α

def dα_dβ_D_collimated(β_value, n):
    s = np.sin(β_value)
    c = np.cos(β_value)

    # Avoid negative due to numerical noise
    root_expression = np.sqrt(np.maximum(n ** 2 - s ** 2, 0.0))

    # NA(β)
    NA = s * (root_expression - c)

    # dNA/dβ
    dNA_dbeta = (
            s * (s - (c * s) / root_expression)
            + c * (root_expression - c)
    )

    # dα/dβ
    denom = np.sqrt(np.maximum(1.0 - NA ** 2, 0.0))

    return dNA_dbeta / denom

back_focal_length = 5e-3
T_c = 3e-3
n_design = 1.45
optical_axis = RIGHT
diameter = 7e-3
back_center = ORIGIN + back_focal_length * optical_axis
aspheric_lens_params = generate_aspheric_lens_params(
                back_focal_length=back_focal_length,
                T_c=T_c,
                n=n_design,
                forward_normal=optical_axis,
                flat_faces_center=back_center,
                diameter=diameter,
                polynomial_degree=8,
                name="Aspheric lens",
            )

surface_0, surface_1 = PhysicalSurface.from_params(aspheric_lens_params)
optical_system = OpticalSystem(surfaces=[surface_0, surface_1])
ray_initial = initialize_rays(n_rays=100, phi_max=0.4)
propagated_ray = optical_system.propagate_ray(ray_initial, propagate_with_first_surface_first=True)
final_normals = surface_1.normal_at_a_point(propagated_ray.origin[-1])
final_k_vectors = propagated_ray.k_vector[-1]
inner_products = np.sum(final_normals * final_k_vectors, axis=1)
betas = np.arccos(np.abs(inner_products))
NAs = propagated_ray.k_vector[0, :, 1]
alphas = np.arcsin(NAs)
betas_analytical = betas_of_NA(NAs=NAs, n=n_design)
alphas_analytical = α_D_collimated(β_value=betas_analytical, n=n_design)
dalphas_d_betas_analytical = dα_dβ_D_collimated(β_value=betas_analytical, n=n_design)
dalphas_d_betas_ray_tracing = np.gradient(alphas, betas)

print("verify output angles are collimated:")
print(np.isclose(final_k_vectors[:, 0], 1))
print("initial angles:")
print(alphas)
print("alphas from analytical formula:")
print(alphas_analytical)
print("initial NAs:")
print(NAs)
print("betas from ray tracing:")
print(betas)
print("betas from analytical formula:")
print(betas_analytical)
print("dα/dβ from analytical formula:")
print(dalphas_d_betas_analytical)
print("dα/dβ from ray tracing:")
print(dalphas_d_betas_ray_tracing)



# %%
fig, ax = plt.subplots()
surface_0.plot(ax=ax, color='blue')
surface_1.plot(ax=ax, color='blue')
plt.plot([propagated_ray.origin[-1, -1, 0], propagated_ray.origin[-1, -1, 0] - final_normals[-1, 0]*1e-3], [propagated_ray.origin[-1, -1, 1], propagated_ray.origin[-1, -1, 1] - final_normals[-1, 1]*1e-3], 'r', label='Final ray positions')
propagated_ray.plot(ax=ax)
ax.set_aspect('equal')
ax.set_xlim(-0.001, 0.01)
ax.grid()
plt.show()

plt.plot(np.sin(betas), betas)

plt.show()