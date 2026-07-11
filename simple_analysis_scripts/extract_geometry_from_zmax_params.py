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

# This should now execute cleanly without the float error
# num_array = get_taylor_numpy_array(20, numeric_params=surf1_params)
# print("Numeric NumPy Array:")
# print("-" * 55)
# print(repr(num_array))

## ATTENTION: the output polynomial here should have it's sign inverse because ZMAX uses the sign convention, and
# my polynomial representation is indifferent to the direction of the beam, so it always has positive curvature.
# num_array = get_taylor_numpy_array(20, numeric_params=surf2_params)
# print("Numeric NumPy Array:")
# print("-" * 55)
# print(repr(num_array))

# %%


params_edmunds = [
    OpticalSurfaceParams(name='aspheric_lens_left', surface_type='aspheric_surface', x=0, y=0, z=0, theta=0,
                         phi=-1e+00 * np.pi, radius=np.nan, curvature_sign=CurvatureSigns.convex, T_c=3.1 / 2e-3,
                         n_inside_or_after=1.583e+00, n_outside_or_before=1e+00, diameter=6.3e-03,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=None,
                                                                beta_surface_absorption=None, kappa_conductivity=None,
                                                                dn_dT=None, nu_poisson_ratio=None,
                                                                alpha_volume_absorption=None,
                                                                intensity_reflectivity=None,
                                                                intensity_transmittance=None, temperature=np.nan),
                         polynomial_coefficients=np.array(
                             [0.00000000e+00, 1.90857799e+02, 3.33213215e+06, 1.08218014e+11, 4.32910754e+15,
                              6.27836273e+20, 2.70839583e+25, 1.72102573e+30, 1.13089247e+35, 7.62167883e+39,
                              5.23938302e+44])),
    OpticalSurfaceParams(name='aspheric_lens_right', surface_type='aspheric_surface', x=3.1e-3, y=0, z=0, theta=0,
                         phi=0, radius=np.nan, curvature_sign=CurvatureSigns.concave, T_c=3.1 / 2e-3,
                         n_inside_or_after=1e+00, n_outside_or_before=1.583e+00, diameter=6.3e-03,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=None,
                                                                beta_surface_absorption=None, kappa_conductivity=None,
                                                                dn_dT=None, nu_poisson_ratio=None,
                                                                alpha_volume_absorption=None,
                                                                intensity_reflectivity=None,
                                                                intensity_transmittance=None, temperature=np.nan),
                         polynomial_coefficients=-np.array(
                             [0.00000000e+00, -3.94470149e+01, 4.40171309e+06, -9.01563757e+10, -1.60183137e+15,
                              -3.23783960e+15, -1.51148859e+19, -7.39192926e+22, -3.73825954e+26, -1.93899408e+30,
                              -1.02584959e+34])),
    OpticalSurfaceParams(name='BK7 window', surface_type=SurfacesTypes.flat_refractive_surface, x=3.1e-3 + 1.483e-3,
                         y=0, z=0, theta=0, phi=0, radius=np.inf, curvature_sign=CurvatureSigns.flat, T_c=np.nan,
                         n_outside_or_before=1, n_inside_or_after=1.5135, material_properties=MaterialProperties(),
                         diameter=6.3e-3, polynomial_coefficients=None)
    ]

optical_system_edmunds = OpticalSystem.from_params(params=params_edmunds, t_is_trivial=True, p_is_trivial=True,
                                                   use_paraxial_ray_tracing=False)
# %%  A Working version with the design wavelength and the glass window:
back_focal_length = back_focal_length_of_lens(R_1=optical_system_edmunds[1].radius,
                                              R_2=-optical_system_edmunds[0].radius, n=optical_system_edmunds[0].n_2,
                                              T_c=optical_system_edmunds[1].center[0] -
                                                  optical_system_edmunds[0].center[0])
initial_ys = np.linspace(0, 0.0015, 10)
initial_xs = np.ones_like(initial_ys) * (-0.01)
initial_positions = np.stack([initial_xs, initial_ys, np.zeros_like(initial_ys)], axis=-1)
ray_0 = Ray(origin=initial_positions, k_vector=RIGHT, n=1)
rays_edmund = optical_system_edmunds.propagate_ray(ray_0, propagate_with_first_surface_first=True)
ax = optical_system_edmunds.plot()
ax.scatter([optical_system_edmunds[1].center[0] + 2.68e-3], [0], color='red', label='Back Focal Point')
rays_edmund.plot(ax=ax)
ax.set_xlim(-0.005, 0.01)
# figure_dir = get_obsidian_save_path(filename="extract_geometry_from_zmax_params.svg")
# plt.savefig(figure_dir)
plt.show()
# %% With the actual wavelength and without the glass window:
optical_system_actual = OpticalSystem.from_params(params=params_edmunds[:2], t_is_trivial=True, p_is_trivial=True,
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

