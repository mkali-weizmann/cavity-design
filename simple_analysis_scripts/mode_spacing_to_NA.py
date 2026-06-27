from copy import deepcopy
from cavity_design import *
from scipy.interpolate import interp1d

N = 100

params = [
          OpticalSurfaceParams(name='Laser Optik Mirror'     ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=5e-03                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None), [
          OpticalSurfaceParams(name='low curvature side - Edmund 4.03mm spherical version',surface_type='curved_refractive_surface'      , x=2.452564065600806e-03   , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=1.267523034472214e-02   , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.574e+00               , n_outside_or_before=1e+00                   , diameter=5.1e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='high curvature side - Edmund 4.03mm spherical version',surface_type='curved_refractive_surface'      , x=5.552564065600805e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.619751468026235e-03   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.574e+00               , diameter=5.1e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)], [
          OpticalSurfaceParams(name='spherical_0'            ,surface_type='curved_refractive_surface'      , x=5.0762e-02              , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=2.542632688301439e-01   , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.51e+00                , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='spherical_1'            ,surface_type='curved_refractive_surface'      , x=5.4472e-02              , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.542632688301439e-01   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.51e+00                , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)],
          OpticalSurfaceParams(name='End mirror'             ,surface_type='curved_mirror'                  , x=3.056414817693546e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2e-01                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=np.nan                  , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)]

cavity = Cavity.from_params(params=params, standing_wave=True,
                                    lambda_0_laser=LAMBDA_0_LASER, power=28e3, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=True, set_central_line=True, set_mode_parameters=True)
short_arm_lengths = np.linspace(7.365e-3, 7.473e-3, N)

# %%

def generate_lens_position_dependencies(short_arm_lengths, plot_dependencies=True):
    NAs = np.zeros(N)
    mode_spacing = np.zeros(N)
    for i, short_arm_length in tqdm(enumerate(short_arm_lengths)):
        cavity_temp = cavity.with_elements_placed(elements=cavity[1], position=short_arm_length*RIGHT, reference_center=cavity[0])
        NAs[i] = cavity_temp.arms[0].mode_parameters.NA[0]
        try:
            mode_spacing[i] = cavity_temp.mode_spacing_transversal_apparent
        except (TypeError, FloatingPointError):
            mode_spacing[i] = np.nan

    if plot_dependencies:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax_twin = ax[0].twinx()
        ax_twin.plot(short_arm_lengths, NAs, label='NA')
        ax_twin.set_ylabel('NA')
        ax[0].plot(short_arm_lengths, mode_spacing / 1e6, label=r'Mode spacing', color='C1')  # use second default color (first is taken by NA)
        ax[0].grid()
        ax[0].set_xlabel("Small arm's length [m]")
        ax[0].legend()

        ax[1].plot(mode_spacing / 1e6, NAs)
        ax[1].set_xlabel(r'Mode spacing [MHz]')
        ax[1].set_ylabel('NA')
        ax[1].set_ylim(0, 0.22)
        ax[1].grid()

        plt.suptitle('Dependencies')
        fig.tight_layout()
        fig.legend()
    plt.show()

    return NAs, mode_spacing

def generate_lens_position_dependencies_output(plot_cavity=True, plot_spectrum=True, plot_dependencies=True):
    if plot_cavity:
        cavity.plot()
        plt.show()
    if plot_spectrum:
        cavity.plot_spectrum(width_over_fsr=0.01)
        plt.show()

    NAs, mode_spacing = generate_lens_position_dependencies(short_arm_lengths,
                                                            plot_dependencies)
    mode_spacing_interp = interp1d(mode_spacing, NAs, fill_value='extrapolate')
    mode_spacing_over_fsr_interp = lambda x: mode_spacing_interp(x) / cavity.fsr
    return mode_spacing_interp, mode_spacing_over_fsr_interp

if __name__ == "__main__":
    mode_spacing_interp, mode_spacing_over_fsr_interp = generate_lens_position_dependencies_output(plot_cavity=True,
                                                                  plot_spectrum=True,
                                                                  plot_dependencies=True
                                                                  )
