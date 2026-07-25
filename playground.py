# %%
from cavity_design import *


u = 50e-6
R_1 = 5e-3
R_2 = 7e-3
L = R_1 + R_2 - u

mirror_1 = SphericalMirror(radius=R_1, origin=ORIGIN, curvature_sign=CurvatureSigns.concave, name='mirror_1', diameter=INCH / 2, outwards_normal=LEFT)
mirror_2 = SphericalMirror(radius=R_2, origin=ORIGIN + u * LEFT, curvature_sign=CurvatureSigns.concave, name='mirror_2', diameter=INCH / 2, outwards_normal=RIGHT)
cavity = Cavity(elements=[mirror_1, mirror_2], use_paraxial_ray_tracing=True, p_is_trivial=True, t_is_trivial=True, lambda_0_laser=LAMBDA_0_LASER)

g_1 = 1 - L / R_1
g_2 = 1 - L / R_2

gouy_phase_term = np.arccos(-np.sqrt(g_1 * g_2))

mode_spacing_analytical = (1-gouy_phase_term / np.pi) * C_LIGHT_SPEED / (2 * L)
mode_spacing_numerical = cavity.mode_spacing_transversal_apparent

w_0_squared_analytical = L * LAMBDA_0_LASER / np.pi * np.sqrt(g_1 * g_2 * (1 - g_1 * g_2) / (g_1 + g_2 -2*g_1 * g_2) ** 2)
w_0_squared_numerical = cavity.mode_parameters[0].w_0**2

w_1_squared_numerical = cavity.arms[0].mode_parameters_on_surface_0.spot_size**2
w_2_squared_numerical = cavity.arms[0].mode_parameters_on_surface_1.spot_size**2

w_1_squared_analytical = L * LAMBDA_0_LASER / np.pi * np.sqrt(g_2 / (g_1 * (1 - g_1 * g_2)))
w_2_squared_analytical = L * LAMBDA_0_LASER / np.pi * np.sqrt(g_1 / (g_2 * (1 - g_1 * g_2)))

z_1_analytical = g_2 * (1-g_1) / (g_1 + g_2 - 2*g_1*g_2) * L
z_2_analytical = g_1 * (1-g_2) / (g_1 + g_2 - 2*g_1*g_2) * L

z_1_numerical = cavity.arms[0].mode_parameters_on_surface_0.z_minus_z_0
z_2_numerical = cavity.arms[0].mode_parameters_on_surface_1.z_minus_z_0
# %%
# print everything:
print(f"g_1: {g_1}, g_2: {g_2}")
print(f"gouy_phase_term: {gouy_phase_term}")
print(f"mode_spacing_analytical: {mode_spacing_analytical}, mode_spacing_numerical: {mode_spacing_numerical}")
print(f"w_0_squared_analytical: {w_0_squared_analytical}, w_0_squared_numerical: {w_0_squared_numerical}")
print(f"w_1_squared_analytical: {w_1_squared_analytical}, w_1_squared_numerical: {w_1_squared_numerical}")
print(f"w_2_squared_analytical: {w_2_squared_analytical}, w_2_squared_numerical: {w_2_squared_numerical}")
print(f"z_1_analytical: {z_1_analytical}, z_1_numerical: {z_1_numerical}")
print(f"z_2_analytical: {z_2_analytical}, z_2_numerical: {z_2_numerical}")
