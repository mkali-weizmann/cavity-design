from matplotlib import use

use("TkAgg")
from cavity_design import *
import sympy as sp


def get_taylor_numpy_array(max_power, numeric_params=None):
    """
    Returns a NumPy array of Taylor coefficients for the Zemax even asphere.
    Index i of the array corresponds to the coefficient for (r^2)^i.

    Parameters:
    max_power (int): The maximum even power (2n) to calculate.
    numeric_params (dict): Optional. Dictionary of numerical values to substitute.
    """
    if max_power % 2 != 0 or max_power < 0:
        raise ValueError("max_power must be a positive even integer.")

    # 1. Define base symbolic variables (Removed real=True to simplify substitution)
    r, c, k = sp.symbols('r c k')

    # Dynamically generate alpha symbols: alpha_1, alpha_2, ..., alpha_n
    n = max_power // 2
    alphas = sp.symbols(f'alpha_1:{n + 1}')

    # Define the base conic formula
    conic_term = (c * r ** 2) / (1 + sp.sqrt(1 - (1 + k) * c ** 2 * r ** 2))

    # Dynamically build the polynomial sum
    poly_term = sum(alpha * r ** (2 * i) for i, alpha in enumerate(alphas, start=1))

    # Total Zemax sag formula
    sag_z = conic_term + poly_term

    # 2. Compute the Taylor series and strictly remove the Big-O term!
    taylor_series = sp.series(sag_z, r, 0, max_power + 2).removeO()

    # Extract coefficients
    coefficients = []
    for power in range(0, max_power + 2, 2):
        coeff = sp.simplify(taylor_series.coeff(r, power))
        coefficients.append(coeff)

    # Substitute numerical values if provided
    if numeric_params:
        # Create a rigid dictionary mapping string names to our actual SymPy symbols
        symbol_dict = {'c': c, 'k': k}
        for i, a_sym in enumerate(alphas, start=1):
            symbol_dict[f'alpha_{i}'] = a_sym

        # 3. Build the substitution dictionary, defaulting missing alphas to 0.0
        safe_subs = {}
        for key_str, sym_obj in symbol_dict.items():
            safe_subs[sym_obj] = numeric_params.get(key_str, 0.0)

        # Evaluate each term into a strict float
        numeric_list = [float(term.subs(safe_subs)) for term in coefficients]
        num_array_mm = np.array(numeric_list)
        indices = np.arange(len(num_array_mm))

        # Calculate 1000^(2i - 1) for each index
        conversion_factors = 1000.0 ** (2 * indices - 1)

        # Multiply and return the new array
        return num_array_mm * conversion_factors
        # ----------------------------------------------

    return np.array(coefficients)


# ==========================================
# Test Run with SURF 1 Parameters
# ==========================================
surf1_params = {
    'c': 3.817155986760137343E-01,
    'k': -0.44495171774960002,
    'alpha_1': 0.0,
    'alpha_2': -0.00052674097012479998,
    'alpha_3': -4.7823832949819998e-05,
    'alpha_4': -3.5582527033080002e-06,
    'alpha_5': 1.8131691173829999e-07
}
surf2_params = {
    'c': -7.889402975752558833E-02,
    'k': 0,
    'alpha_1': 0.0,
    'alpha_2': 0.0044630952841129999,
    'alpha_3': -8.9965346051199994e-05,
    'alpha_4': -1.6010882291619999e-06,
}

surf1_thorlabs_354240_params = {
    'c': 2.156044124736639089E-01,
    'k': -0.92552100000000004,
    'alpha_1': 0.0,
    'alpha_2': 0.0004789735,
    'alpha_3': 4.0496920000000002e-06,
    'alpha_4': 3.1281810000000001e-08,
    'alpha_5': -6.4986990000000002e-10
}


# zmax_37104.zmx
surf1_params_37104 = {
    'c': 2.156044124736639089E-01,
    'k': -0.92552199999999996,
    'alpha_1': 0.0,
    'alpha_2': 0.0004789735,
    'alpha_3': 4.0496920000000002e-06,
    'alpha_4': 3.1281810000000001e-08,
    'alpha_5': -6.4986990000000002e-10,
    'alpha_6': 0.0,
    'alpha_7': 0.0,
    'alpha_8': 0.0
}

# zmax_87127.zmx
surf1_params_87127 = {
    'c': 2.020869508567745831E-01,
    'k': -1.1166180220269999,
    'alpha_1': 0.0,
    'alpha_2': 0.00060044508716450003,
    'alpha_3': 1.3641596756549999e-06,
    'alpha_4': -6.9075167114429994e-08,
    'alpha_5': 0.0,
    'alpha_6': 0.0,
    'alpha_7': 0.0,
    'alpha_8': 0.0
}

# Eksma 117-1220 fused-silica aspheric lens
surf1_params_eksma = {
    'c': 1.102171277416510442E-01,
    'k': -0.67000000000000004,
    'alpha_1': 0.0,
    'alpha_2': 1.7503618979310571e-05,
    'alpha_3': 5.0392717735941379e-08,
    'alpha_4': 1.1411005348584909e-10,
    'alpha_5': 0.0,
    'alpha_6': 0.0,
    'alpha_7': 0.0,
    'alpha_8': 0.0
}

## ATTENTION: the output polynomial for second surfaceshould have it's sign inverse because ZMAX uses the sign convention, and
# my polynomial representation is indifferent to the direction of the beam, so it always has positive curvature.
num_array = get_taylor_numpy_array(20, numeric_params=surf1_params_eksma)
print("Numeric NumPy Array:")
print("-" * 55)
print(repr(num_array))

# %%
optical_system_edmunds = OpticalSystem(
    elements=[
        AsphericRefractiveSurface(name='aspheric_lens_left', center=np.array([0.0, 0.0, 0.0]), outwards_normal=np.array([-1.0, 0.0, 0.0]), polynomial_coefficients=np.array([0.0, 190.857799, 3332132.15, 108218014000.0, 4329107540000000.0, 6.27836273e+20, 2.70839583e+25, 1.72102573e+30, 1.13089247e+35, 7.62167883e+39, 5.23938302e+44]), curvature_sign=1, n_1=1.0, n_2=1.583, diameter=0.0063, material_properties=MaterialProperties(temperature=np.nan)),
        AsphericRefractiveSurface(name='aspheric_lens_right', center=np.array([0.0031, 0.0, 0.0]), outwards_normal=np.array([1.0, 0.0, 0.0]), polynomial_coefficients=np.array([-0.0, 39.4470149, -4401713.09, 90156375700.0, 1601831370000000.0, 3237839600000000.0, 1.51148859e+19, 7.39192926e+22, 3.73825954e+26, 1.93899408e+30, 1.02584959e+34]), curvature_sign=-1, n_1=1.583, n_2=1.0, diameter=0.0063, material_properties=MaterialProperties(temperature=np.nan)),
        FlatRefractiveSurface(name='BK7 window', outwards_normal=np.array([1.0, 0.0, 0.0]), center=np.array([0.004583, 0.0, 0.0]), n_1=1, n_2=1.5135, diameter=0.0063, thermal_properties=MaterialProperties(temperature=np.nan)),
    ],
    use_paraxial_ray_tracing=False,     lambda_0_laser=1.064e-06,     t_is_trivial=True,     p_is_trivial=True,
)

# %%  A Working version with the design wavelength and the glass window:
back_focal_length = back_focal_length_of_lens_formula(R_1=optical_system_edmunds[1].radius,
                                                      R_2=-optical_system_edmunds[0].radius,
                                                      n=optical_system_edmunds[0].n_2,
                                                      T_c=optical_system_edmunds[1].center[0] -
                                                          optical_system_edmunds[0].center[0])
initial_ys = np.linspace(0, 0.0015, 10)
initial_xs = np.ones_like(initial_ys) * (-0.01)
initial_positions = np.stack([initial_xs, initial_ys, np.zeros_like(initial_ys)], axis=-1)
ray_0 = Ray(origin=initial_positions, k_vector=RIGHT, n=1)
rays_thorlbs = optical_system_edmunds.propagate_ray(ray_0, propagate_with_first_surface_first=True)
ax = optical_system_edmunds.plot()
ax.scatter([optical_system_edmunds[1].center[0] + 2.68e-3], [0], color='red', label='Back Focal Point')
rays_thorlbs.plot(ax=ax)
ax.set_xlim(-0.005, 0.01)
# figure_dir = get_obsidian_save_path(filename="extract_geometry_from_zmax_params.svg")
# plt.savefig(figure_dir)
plt.show()
# %% With the actual wavelength and without the glass window:
optical_system_actual = OpticalSystem.from_params(params=optical_system_edmunds.elements[:2], t_is_trivial=True, p_is_trivial=True,
                                                  use_paraxial_ray_tracing=False)
optical_system_actual[0].n_2 = 1.574
optical_system_actual[1].n_1 = 1.574
rays_actual = optical_system_actual.propagate_ray(ray_0, propagate_with_first_surface_first=True)
ax = optical_system_actual.plot()
rays_actual.plot(ax=ax)
ax.set_xlim(-0.005, 0.01)
# figure_dir = get_obsidian_save_path(filename="extract_geometry_from_zmax_params-actual.svg")
# plt.savefig(figure_dir)
plt.title("Actual wavelength (1064nm) and actual refractive index (1.574) without the glass window")
plt.show()
# %% With an equivalent spherical lens:
optical_system_spherical = OpticalSystem(elements=[
    CurvedRefractiveSurface(name="spherical - left",
                            radius=optical_system_actual[0].radius, outwards_normal=LEFT,
                            center=optical_system_actual[0].center, n_1=1, n_2=optical_system_actual[0].n_2, diameter=optical_system_actual[0].diameter, curvature_sign=CurvatureSigns.convex),
    CurvedRefractiveSurface(name="spherical - right",
                            radius=optical_system_actual[1].radius, outwards_normal=RIGHT,
                            center=optical_system_actual[1].center, n_1=optical_system_actual[1].n_1, n_2=1, diameter=optical_system_actual[1].diameter, curvature_sign=CurvatureSigns.concave),
], use_paraxial_ray_tracing=False, p_is_trivial=True, t_is_trivial=True)
rays_spherical = optical_system_spherical.propagate_ray(ray_0, propagate_with_first_surface_first=True)
ax = optical_system_spherical.plot()
rays_spherical.plot(ax=ax)
ax.set_xlim(-0.005, 0.01)
plt.title("Equivalent spherical lens with the same radii of curvature and thickness as the aspheric lens")
plt.show()

# %% Thorlabs aspheric:
optical_system_thorlabs = OpticalSystem(
    elements=[
        AsphericRefractiveSurface(name='aspheric_lens_convex', center=np.array([0.0, 0.0, 0.0]), outwards_normal=np.array([-1.0, 0.0, 0.0]), polynomial_coefficients=np.array([0.0, 107.802206, 572281.05, 4211215570.0, 31631324400000.0, -6.49022844e+17, 2.19949315e+18, 5.98323954e+21, 1.68309599e+25, 4.85597863e+28, 1.42904143e+32]), curvature_sign=1, n_1=1.0, n_2=1.584, diameter=0.0082, material_properties=MaterialProperties(temperature=np.nan)),
        FlatRefractiveSurface(name='aspheric_lens_flat', outwards_normal=np.array([1.0, 0.0, 0.0]), center=np.array([0.003434, 0.0, 0.0]), n_1=1.584, n_2=1, diameter=0.0082, thermal_properties=MaterialProperties(temperature=np.nan)),
        FlatRefractiveSurface(name='window_left', outwards_normal=np.array([-1.0, 0.0, 0.0]), center=np.array([0.008434, 0.0, 0.0]), n_1=1, n_2=1.5135, diameter=0.0082, thermal_properties=MaterialProperties(temperature=np.nan)),
        FlatRefractiveSurface(name='window_right', outwards_normal=np.array([-1.0, 0.0, 0.0]), center=np.array([0.008684, 0.0, 0.0]), n_1=1.5135, n_2=1, diameter=0.0082, thermal_properties=MaterialProperties(temperature=np.nan)),
    ],
    use_paraxial_ray_tracing=False,     lambda_0_laser=1.064e-06,     t_is_trivial=True,     p_is_trivial=True,
)
# %%
# optical_system_thorlabs.plot()
# plt.show()
# plt.close('all')
back_focal_length = back_focal_length_of_lens_formula(R_1=optical_system_thorlabs[1].radius,
                                                      R_2=-optical_system_thorlabs[0].radius,
                                                      n=optical_system_thorlabs[0].n_2,
                                                      T_c=optical_system_thorlabs[1].center[0] -
                                                          optical_system_thorlabs[0].center[0])
initial_ys = np.linspace(0, 0.0015, 10)
initial_xs = np.ones_like(initial_ys) * (-0.01)
initial_positions = np.stack([initial_xs, initial_ys, np.zeros_like(initial_ys)], axis=-1)
ray_0 = Ray(origin=initial_positions, k_vector=RIGHT, n=1)
rays_thorlbs = optical_system_thorlabs.propagate_ray(ray_0, propagate_with_first_surface_first=True)
ax = optical_system_thorlabs.plot()
ax.scatter([optical_system_thorlabs[1].center[0] + back_focal_length], [0], color='red', label='Back Focal Point')
rays_thorlbs.plot(ax=ax)
ax.set_xlim(-0.005, 0.01)
# figure_dir = get_obsidian_save_path(filename="extract_geometry_from_zmax_params.svg")
# plt.savefig(figure_dir)
plt.show()
# %% With the actual wavelength and without the glass window:
optical_system_actual = OpticalSystem(elements=optical_system_thorlabs[:2], t_is_trivial=True, p_is_trivial=True,
                                                  use_paraxial_ray_tracing=False)
optical_system_actual[0].n_2 = 1.577
optical_system_actual[1].n_1 = 1.577
rays_actual = optical_system_actual.propagate_ray(ray_0, propagate_with_first_surface_first=True)
ax = optical_system_actual.plot()
rays_actual.plot(ax=ax)
ax.set_xlim(-0.005, 0.01)
# figure_dir = get_obsidian_save_path(filename="extract_geometry_from_zmax_params-actual - thorlabs.svg")
# plt.savefig(figure_dir)
plt.title("Actual wavelength (1064nm) and actual refractive index (1.577) without the glass window")
plt.show()
# %% With an equivalent spherical lens:
optical_system_spherical = OpticalSystem(elements=[
    CurvedRefractiveSurface(name="spherical - left",
                            radius=optical_system_actual[0].radius, outwards_normal=LEFT,
                            center=optical_system_actual[0].center, n_1=1, n_2=optical_system_actual[0].n_2, diameter=optical_system_actual[0].diameter, curvature_sign=CurvatureSigns.convex),
    FlatRefractiveSurface(name="spherical - right",
                            outwards_normal=RIGHT,
                            center=optical_system_actual[1].center, n_1=optical_system_actual[1].n_1, n_2=1, diameter=optical_system_actual[1].diameter),
], use_paraxial_ray_tracing=False, p_is_trivial=True, t_is_trivial=True)
rays_spherical = optical_system_spherical.propagate_ray(ray_0, propagate_with_first_surface_first=True)
ax = optical_system_spherical.plot()
rays_spherical.plot(ax=ax)
ax.set_xlim(-0.005, 0.01)
plt.title("Equivalent spherical lens with the same radii of curvature and thickness as the aspheric lens")
plt.show()


# %% Edmund 37104:
optical_system_37104 = OpticalSystem(
    elements=[
        AsphericRefractiveSurface(name='aspheric_lens_convex', center=np.array([0.0, 0.0, 0.0]), outwards_normal=np.array([-1.0, 0.0, 0.0]), polynomial_coefficients=np.array([0.0, 107.802206, 572279.797, 4211211240.0, 31631310300000.0, -6.49022889e+17, 2.1993455e+18, 5.98275755e+21, 1.68293781e+25, 4.85545706e+28, 1.42886875e+32]), curvature_sign=1, n_1=1.0, n_2=1.58, diameter=0.0082, material_properties=MaterialProperties(temperature=np.nan)),
        FlatRefractiveSurface(name='aspheric_lens_flat', outwards_normal=np.array([1.0, 0.0, 0.0]), center=np.array([0.003434, 0.0, 0.0]), n_1=1.58, n_2=1, diameter=0.0082, thermal_properties=MaterialProperties(temperature=np.nan)),
        FlatRefractiveSurface(name='window_left', outwards_normal=np.array([-1.0, 0.0, 0.0]), center=np.array([0.008345, 0.0, 0.0]), n_1=1, n_2=1.5135, diameter=0.0082, thermal_properties=MaterialProperties(temperature=np.nan)),
        FlatRefractiveSurface(name='window_right', outwards_normal=np.array([-1.0, 0.0, 0.0]), center=np.array([0.008595, 0.0, 0.0]), n_1=1.5135, n_2=1, diameter=0.0082, thermal_properties=MaterialProperties(temperature=np.nan)),
    ],
    use_paraxial_ray_tracing=False,     lambda_0_laser=1.064e-06,     t_is_trivial=True,     p_is_trivial=True,
)
# %%
# optical_system_37104.plot()
# plt.show()
# plt.close('all')
back_focal_length = back_focal_length_of_lens_formula(R_1=optical_system_37104[1].radius,
                                                      R_2=-optical_system_37104[0].radius,
                                                      n=optical_system_37104[0].n_2,
                                                      T_c=optical_system_37104[1].center[0] -
                                                          optical_system_37104[0].center[0])
initial_ys = np.linspace(0, 0.0015, 10)
initial_xs = np.ones_like(initial_ys) * (-0.01)
initial_positions = np.stack([initial_xs, initial_ys, np.zeros_like(initial_ys)], axis=-1)
ray_0 = Ray(origin=initial_positions, k_vector=RIGHT, n=1)
rays_thorlbs = optical_system_37104.propagate_ray(ray_0, propagate_with_first_surface_first=True)
ax = optical_system_37104.plot()
ax.scatter([optical_system_37104[1].center[0] + back_focal_length], [0], color='red', label='Back Focal Point')
rays_thorlbs.plot(ax=ax)
ax.set_xlim(-0.005, 0.01)
# figure_dir = get_obsidian_save_path(filename="extract_geometry_from_zmax_params.svg")
# plt.savefig(figure_dir)
plt.show()
# %% With the actual wavelength and without the glass window:
optical_system_actual = OpticalSystem(elements=optical_system_37104.elements[:2], t_is_trivial=True, p_is_trivial=True,
                                                  use_paraxial_ray_tracing=False)
optical_system_actual[0].n_2 = 1.574
optical_system_actual[1].n_1 = 1.574
rays_actual = optical_system_actual.propagate_ray(ray_0, propagate_with_first_surface_first=True)
ax = optical_system_actual.plot()
rays_actual.plot(ax=ax)
ax.set_xlim(-0.005, 0.01)
# figure_dir = get_obsidian_save_path(filename="extract_geometry_from_zmax_params-actual - 37104.svg")
# plt.savefig(figure_dir)
plt.title("Actual wavelength (1064nm) and actual refractive index (1.577) without the glass window")
plt.show()
# %% With an equivalent spherical lens:
optical_system_spherical = OpticalSystem(elements=[
    CurvedRefractiveSurface(name="spherical - left",
                            radius=optical_system_actual[0].radius, outwards_normal=LEFT,
                            center=optical_system_actual[0].center, n_1=1, n_2=optical_system_actual[0].n_2, diameter=optical_system_actual[0].diameter, curvature_sign=CurvatureSigns.convex),
    FlatRefractiveSurface(name="spherical - right",
                            outwards_normal=RIGHT,
                            center=optical_system_actual[1].center, n_1=optical_system_actual[1].n_1, n_2=1, diameter=optical_system_actual[1].diameter),
], use_paraxial_ray_tracing=False, p_is_trivial=True, t_is_trivial=True)
rays_spherical = optical_system_spherical.propagate_ray(ray_0, propagate_with_first_surface_first=True)
ax = optical_system_spherical.plot()
rays_spherical.plot(ax=ax)
ax.set_xlim(-0.005, 0.01)
plt.title("Equivalent spherical lens with the same radii of curvature and thickness as the aspheric lens")
plt.show()
