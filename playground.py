from matplotlib import use
use('TkAgg')
from simple_analysis_scripts.potential_analysis.analyze_potential import *
from scipy.optimize import minimize_scalar



params = [
          OpticalElementParams(name='LaserOptik mirror'      ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='spherical_lens'         ,surface_type='thick_lens'                     , x=6.593276315694677e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=-5.488e-03              , curvature_sign=CurvatureSigns.convex, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='Negative Lens'          ,surface_type='thick_lens'                     , x=3.604751750861879e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.793113337652691e-02   , r_2=1.501501501501501e-02   , curvature_sign=CurvatureSigns.concave, T_c=3.45e-03                , n_inside_or_after=1.45e+00                , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='End mirror'             ,surface_type='curved_mirror'                  , x=4.050459581077056e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=1.000000000464597e-01   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=np.nan                  , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None)]

cavity = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=LAMBDA_0_LASER,
                            set_central_line=True,
                            set_mode_parameters=True,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4,
                            use_paraxial_ray_tracing=False,
                            debug_printing_level=1,
                            )
# %%
t, p = 0, 0.001
p_1 = cavity.surfaces[-1].parameterization(t, p)
initial_guess_direction = -cavity.surfaces[-1].normal_at_a_point(point=p_1)
initial_guess_theta, initial_guess_phi = angles_of_unit_vector(initial_guess_direction)
initial_guess_ray = Ray(origin=p_1, k_vector=initial_guess_direction)
optical_system_reduced = OpticalSystem(surfaces=cavity.surfaces_ordered[len(cavity.surfaces_ordered) // 2:],
                                       use_paraxial_ray_tracing=False,
                                       p_is_trivial=True,
                                       t_is_trivial=True,)

# Reuse a single figure to avoid creating a new one each call
fig, ax = plt.subplots()

phi = initial_guess_phi
def f_for_extremum(theta, phi):
    k_vector = unit_vector_of_angles(theta=theta, phi=phi)
    ray = Ray(origin=p_1, k_vector=k_vector)
    propagated_ray = optical_system_reduced.propagate_ray(ray=ray, propagate_with_first_surface_first=False)
    total_path_length = propagated_ray.cumulative_optical_path_length[-2]
    last_inner_product = propagated_ray.k_vector[-2] @ propagated_ray.k_vector[-1]
    if np.isnan(total_path_length):
        total_path_length = np.inf
        last_inner_product = np.inf
    ax.clear()
    optical_system_reduced.plot(ax=ax)
    propagated_ray.plot(ax=ax)
    ax.set_title(f'$\phi={phi:.10f}$\nOptical path length: {total_path_length:.10f} m\n last_inner product: {last_inner_product:.10f}')
    fig.canvas.draw_idle()
    plt.pause(0.001)
    return total_path_length

def f_for_extremum_1d(phi):
    return f_for_extremum(theta=initial_guess_theta, phi=phi)

optical_path_length = f_for_extremum(f_for_extremum_1d)

# find phi that minimizes the optical path length

result = minimize_scalar(f_for_extremum_1d, bounds=(initial_guess_phi - np.pi / 1000, initial_guess_phi + np.pi / 1000), tol=1e-10)
optimal_phi = result.x
optimal_optical_path_length = result.fun
# %%
phis_dummy = np.linspace(initial_guess_phi - np.pi / 1000, initial_guess_phi + np.pi / 1000, 100)
propagated_ray_dummy = optical_system_reduced.propagate_ray(ray=Ray(origin=p_1, k_vector=unit_vector_of_angles(theta=initial_guess_theta, phi=phis_dummy)), propagate_with_first_surface_first=False)
fig, ax = plt.subplots()
optical_system_reduced.plot(ax=ax)
propagated_ray_dummy[:-1].plot(ax=ax, linewidth=0.5)
ax.set_title(f'Optimal $\phi$: {optimal_phi:.10f}\nOptimal optical path length: {optimal_optical_path_length:.10f}m')
plt.show()

# jacobian_phi =
