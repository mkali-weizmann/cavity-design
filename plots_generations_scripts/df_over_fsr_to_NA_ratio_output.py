from copy import deepcopy
from cavity import *
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

N = 500
params_actual_cavity = [
    OpticalElementParams(name='Small Mirror', surface_type='curved_mirror', x=2.399e-02, y=-1.355252715606881e-20, z=0,
                         theta=-4.313903376550994e-21 * np.pi, phi=-4.313903376550994e-21 * np.pi, r_1=2.4e-02,
                         r_2=np.nan, curvature_sign=CurvatureSigns.concave, T_c=np.nan, n_inside_or_after=1e+00,
                         n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.31e+00, dn_dT=None,
                                                                nu_poisson_ratio=1.7e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan)),
    OpticalElementParams(name='Lens', surface_type='thick_lens', x=-2.1877586e-02, y=-1.355252715606881e-20, z=0,
                         theta=-4.313903376550994e-21 * np.pi, phi=-1e+00 * np.pi, r_1=1.77e-02, r_2=1.77e-02,
                         curvature_sign=CurvatureSigns.concave, T_c=4.2e-03, n_inside_or_after=1.45e+00,
                         n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=1.507e+00, alpha_expansion=7.1e-06,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.114e+00, dn_dT=None,
                                                                nu_poisson_ratio=2.06e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan)),
    OpticalElementParams(name='Big Mirror', surface_type='curved_mirror', x=-8.3601e-01, y=-1.355252715606881e-20, z=0,
                         theta=-4.313903376550994e-21 * np.pi, phi=-1e+00 * np.pi, r_1=5e-01, r_2=np.nan,
                         curvature_sign=CurvatureSigns.concave, T_c=np.nan, n_inside_or_after=1e+00,
                         n_outside_or_before=1e+00,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.31e+00, dn_dT=None,
                                                                nu_poisson_ratio=1.7e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan))]

delta_xs_lens_actual = double_exponential_spacing(-0.000575, 0.000328, N)


def generate_lens_position_dependencies(params, delta_xs_lens, plot_dependencies=True):
    NAs = np.zeros(N)
    df_over_FSR = np.zeros(N)
    df_over_FSR_theory = np.zeros(N)
    Ls = np.zeros(N)
    for i, delta_x_lens in enumerate(delta_xs_lens):
        if i % 100 == 0:
            print(f"\r{i}/{N}", end="\r")

        params_copy = deepcopy(params)
        params_copy[1].x += delta_x_lens
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
        Ls[i] = np.linalg.norm(params_copy[1].x - params_copy[0].x)
        try:
            df_over_FSR_temp = cavity.delta_f_frequency_transversal_modes / cavity.free_spectral_range
            # df_over_FSR_temp = #np.abs(np.mod(df_over_FSR_temp + 0.5, 1) - 0.5)
            df_over_FSR[i] = df_over_FSR_temp

        except (TypeError, FloatingPointError):
            df_over_FSR[i] = np.nan
            df_over_FSR_theory[i] = np.nan
    print(f"\r{N}/{N}", end="\r")

    if plot_dependencies:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(Ls, NAs, label='NA')
        ax[0].plot(Ls, df_over_FSR, label=r'$\frac{df}{\text{FSR}}$ - simulation)')  # '
        ax[0].grid()
        # ax2 = ax[0].twinx()
        # ax2.plot(Ls, delta_xs_lens, label=r'$\Delta{x,\text{lens}}$', color='red')

        ax[0].set_xlabel("Small arm's length [m]")
        # ax[0].set_xlim(0.048, 0.051)
        ax[0].legend()

        ax[1].plot(df_over_FSR, NAs)
        ax[1].set_xlabel(r'$\frac{df}{\text{FSR}}$')
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
        cavity = Cavity.from_params(params=params_actual_cavity,
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

    NAs, df_over_FSR, Ls = generate_lens_position_dependencies(params_actual_cavity,
                                                               delta_xs_lens_actual,
                                                               plot_dependencies)
    df_over_FSR_interp = interp1d(df_over_FSR, NAs, fill_value='extrapolate')
    return df_over_FSR_interp, NAs, df_over_FSR, Ls

if __name__ == "__main__":
    df_over_FSR_interp, NAs, df_over_FSR, Ls = generate_lens_position_dependencies_output(plot_cavity=False,
                                                                  plot_spectrum=False,
                                                                  plot_dependencies=True
                                                                  )
