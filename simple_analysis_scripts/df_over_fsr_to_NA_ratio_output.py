from copy import deepcopy
from cavity_design import *
from scipy.interpolate import interp1d


def double_exponential_spacing(v_min, v_max, num_points=100):
    diff = v_max - v_min
    dummy_array = (np.cosh(np.linspace(0, np.log(2 + np.sqrt(3)), num_points // 2, endpoint=True)) - 1) * diff / 2
    left = v_min + dummy_array
    right = v_max - dummy_array[::-1]
    if num_points % 2 == 0:
        return np.concatenate([left, right])
    else:
        return np.concatenate([left, [v_max], right[1:]])

N = 100
params = [
          OpticalSurfaceParams(name='Laser Optik Mirror'     ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=5e-03                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None), [
          OpticalSurfaceParams(name='aspheric_lens_automatic - flat side',surface_type='flat_refractive_surface'        , x=4.08372545116322e-03    , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , radius=0                       , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1.574e+00               , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='aspheric_lens_automatic - curved side',surface_type='aspheric_surface'               , x=7.18372545116322e-03    , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.704137204127337e-03   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.574e+00               , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=np.array([ 3.78495078e-11,  1.84901860e+02,  3.05345512e+06,  7.70343165e+10, 2.08157839e+15, -2.20570527e+20,  1.08097479e+26, -3.70747171e+31, 7.97890251e+36, -1.09308973e+42,  9.11551800e+46, -4.22614376e+51, 8.35809886e+55]))
         ], [
          OpticalSurfaceParams(name='spherical_0'            ,surface_type='curved_refractive_surface'      , x=3.718372545116322e-02   , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=2.542632688301439e-01   , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.51e+00                , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='spherical_1'            ,surface_type='curved_refractive_surface'      , x=4.153372545116321e-02   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.542632688301439e-01   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.51e+00                , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)],
          OpticalSurfaceParams(name='End mirror'             ,surface_type='curved_mirror'                  , x=2.235282389335875e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2e-01                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=2.54e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)]

cavity = Cavity.from_params(params=params, standing_wave=True,
                                    lambda_0_laser=LAMBDA_0_LASER, power=28e3, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=True, set_central_line=True, set_mode_parameters=True)
cavity.plot()
plt.title(f"NA_short = {cavity.mode_parameters[0].NA[0]:.2e}, NA_long = {cavity.mode_parameters[2].NA[0]:.2e}\nlength short = {cavity.central_line[0].length*1000:.2f}mm, length long = {cavity.central_line[2].length*1000:.2f}mm")
plt.show()
delta_xs_lens_actual = double_exponential_spacing(-0.001, 0.001, N)
# %%

def generate_lens_position_dependencies(params, delta_xs_lens, plot_dependencies=True):
    NAs = np.zeros(N)
    df_over_FSR = np.zeros(N)
    df_over_FSR_theory = np.zeros(N)
    Ls = np.zeros(N)
    for i, delta_x_lens in enumerate(delta_xs_lens):
        if i % 100 == 0:
            print(f"\r{i}/{N}", end="\r")

        params_copy = deepcopy(params)
        params_copy[1][0].x += delta_x_lens
        params_copy[1][1].x += delta_x_lens
        cavity = Cavity.from_params(params=params_copy,
                                    standing_wave=True,
                                    lambda_0_laser=LAMBDA_0_LASER,
                                    set_central_line=True,
                                    set_mode_parameters=True,
                                    set_initial_surface=False,
                                    t_is_trivial=True,
                                    p_is_trivial=True,
                                    power=2e4,
                                    use_paraxial_ray_tracing=True,
                                    debug_printing_level=1,
                                    )

        NAs[i] = cavity.arms[0].mode_parameters.NA[0]
        Ls[i] = np.linalg.norm(params_copy[1][0].x - params_copy[0].x)
        try:
            df_over_FSR_temp = cavity.delta_f_frequency_transversal_modes / cavity.free_spectral_range
            df_over_FSR_temp = np.abs(np.mod(df_over_FSR_temp + 0.5, 1) - 0.5)
            df_over_FSR[i] = df_over_FSR_temp

        except (TypeError, FloatingPointError):
            df_over_FSR[i] = np.nan
            df_over_FSR_theory[i] = np.nan
    print(f"\r{N}/{N}", end="\r")

    if plot_dependencies:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(Ls, NAs, label='NA')
        ax[0].plot(Ls, df_over_FSR, label=r'$\frac{df}{FSR}$ - simulation)')  # '
        ax[0].grid()
        # ax2 = ax[0].twinx()
        # ax2.plot(Ls, delta_xs_lens, label=r'$\Delta{x,\text{lens}}$', color='red')

        ax[0].set_xlabel("Small arm's length [m]")
        # ax[0].set_xlim(0.048, 0.051)
        ax[0].legend()

        ax[1].plot(df_over_FSR, NAs)
        ax[1].set_xlabel(r'$\frac{df}{FSR}$')
        ax[1].set_ylabel('NA')
        # ax[1].set_xlim(0, 0.12)
        ax[1].set_ylim(0, 0.22)
        ax[1].grid()
        plt.suptitle('Dependencies')
        fig.tight_layout()
        # fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax[0].transAxes)
        fig.legend()
    plt.show()

    return NAs, df_over_FSR, Ls


def generate_lens_position_dependencies_output(plot_cavity=True, plot_spectrum=True, plot_dependencies=True):
    if plot_cavity or plot_spectrum:
        cavity = Cavity.from_params(params=params,
                                    standing_wave=True,
                                    lambda_0_laser=LAMBDA_0_LASER,
                                    set_central_line=True,
                                    set_mode_parameters=True,
                                    set_initial_surface=False,
                                    t_is_trivial=True,
                                    p_is_trivial=True,
                                    power=2e4,
                                    use_paraxial_ray_tracing=True,
                                    debug_printing_level=1,
                                    )
        if plot_cavity:
            plot_mirror_lens_mirror_cavity_analysis(cavity, CA=12.5e-3)
            plt.ylim(-0.003, 0.003)
            plt.xlim(-0.9, 0.05)
            plt.show()
        if plot_spectrum:
            cavity.plot_spectrum(width_over_fsr=0.01)
            plt.show()

    NAs, df_over_FSR, Ls = generate_lens_position_dependencies(params,
                                                               delta_xs_lens_actual,
                                                               plot_dependencies)
    df_over_FSR_interp = interp1d(df_over_FSR, NAs, fill_value='extrapolate')
    return df_over_FSR_interp, NAs, df_over_FSR, Ls

if __name__ == "__main__":
    df_over_FSR_interp, NAs, df_over_FSR, Ls = generate_lens_position_dependencies_output(plot_cavity=False,
                                                                  plot_spectrum=False,
                                                                  plot_dependencies=True
                                                                  )
