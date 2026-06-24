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
          OpticalSurfaceParams(name='low curvature side - Edmund 4.03mm spherical version',surface_type='curved_refractive_surface'      , x=2.452564065600806e-03   , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=1.267523034472214e-02   , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.574e+00               , n_outside_or_before=1e+00                   , diameter=5.1e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='high curvature side - Edmund 4.03mm spherical version',surface_type='curved_refractive_surface'      , x=5.552564065600805e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.619751468026235e-03   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.574e+00               , diameter=5.1e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)], [
          OpticalSurfaceParams(name='spherical_0'            ,surface_type='curved_refractive_surface'      , x=5.0762e-02              , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=2.542632688301439e-01   , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.51e+00                , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='spherical_1'            ,surface_type='curved_refractive_surface'      , x=5.4472e-02              , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.542632688301439e-01   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.51e+00                , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)],
          OpticalSurfaceParams(name='End mirror'             ,surface_type='curved_mirror'                  , x=3.056414817693546e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2e-01                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=np.nan                  , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)]

cavity = Cavity.from_params(params=params, standing_wave=True,
                                    lambda_0_laser=LAMBDA_0_LASER, power=28e3, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=True, set_central_line=True, set_mode_parameters=True)
# cavity.plot()
# plt.title(f"NA_short = {cavity.mode_parameters[0].NA[0]:.2e}, NA_long = {cavity.mode_parameters[2].NA[0]:.2e}\nlength short = {cavity.central_line[0].length*1000:.2f}mm, length long = {cavity.central_line[2].length*1000:.2f}mm")
# plt.show()
short_arm_lengths = np.linspace(7.365e-3, 7.473e-3, N)

# %%

def generate_lens_position_dependencies(params, short_arm_lengths, plot_dependencies=True):
    NAs = np.zeros(N)
    df_over_FSR = np.zeros(N)
    df_over_FSR_theory = np.zeros(N)
    Ls = np.zeros(N)
    T_c_lens = params[1][1].x - params[1][0].x
    for i, short_arm_length in enumerate(short_arm_lengths):
        if i % 100 == 0:
            print(f"\r{i}/{N}", end="\r")

        params_copy = deepcopy(params)
        params_copy[1][0].x = params[0].x + short_arm_length
        params_copy[1][1].x = params[0].x + short_arm_length + T_c_lens
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
            df_over_FSR_temp = cavity.delta_f_frequency_transversal / cavity.free_spectral_range
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
                                                               short_arm_lengths,
                                                               plot_dependencies)
    df_over_FSR_interp = interp1d(df_over_FSR, NAs, fill_value='extrapolate')
    return df_over_FSR_interp, NAs, df_over_FSR, Ls

if __name__ == "__main__":
    df_over_FSR_interp, NAs, df_over_FSR, Ls = generate_lens_position_dependencies_output(plot_cavity=False,
                                                                  plot_spectrum=False,
                                                                  plot_dependencies=True
                                                                  )
