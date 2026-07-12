from copy import deepcopy
from cavity_design import *
from scipy.interpolate import interp1d

N = 100

cavity = Cavity(elements=[LASER_OPTIK_MIRROR,
                          EDMUND_4p03MM_ASPHERIC,
                          THOLABS_100MM_PLANO_CONVEX_LENS,
                          COASTLINE_20CM_MIRROR],
                use_paraxial_ray_tracing=True, p_is_trivial=True, t_is_trivial=True, lambda_0_laser=LAMBDA_0_LASER)
aspheric_BFL = back_focal_length_of_lens_object(lens_object=EDMUND_4p03MM_ASPHERIC)
collimation_point = cavity[0].radius + aspheric_BFL

# %%

def generate_lens_position_dependencies(short_arm_lengths: np.ndarray,
                                        mid_arm_length: float,
                                        long_arm_length: float,
                                        plot_dependencies=True):
    NAs = np.zeros(N)
    mode_spacing = np.zeros(N)
    # nominal_positions:
    cavity.place_element(element=cavity[1], position=collimation_point * RIGHT, reference_center=cavity[0],
                         recalculate_optic=False)
    cavity.place_element(element=cavity[2], position=mid_arm_length * RIGHT, reference_center=cavity[1],
                         recalculate_optic=False)
    cavity.place_element(element=cavity[3], position=long_arm_length * RIGHT, reference_center=cavity[2],
                         recalculate_optic=False)

    for i, short_arm_length in tqdm(enumerate(short_arm_lengths)):
        cavity.place_element(element=cavity[1], position=short_arm_length * RIGHT, reference_center=cavity[0],
                             recalculate_optic=True)
        NAs[i] = cavity.arms[0].mode_parameters.NA[0]
        try:
            mode_spacing[i] = cavity.mode_spacing_transversal_apparent
        except (TypeError, FloatingPointError):
            mode_spacing[i] = np.nan

    if plot_dependencies:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax_twin = ax[0].twinx()
        ax_twin.plot(short_arm_lengths, NAs, label='NA')
        ax_twin.set_ylabel('NA')
        ax[0].plot(short_arm_lengths, mode_spacing / 1e6, label=r'Mode spacing',
                   color='C1')  # use second default color (first is taken by NA)
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


def generate_lens_position_dependencies_output(short_arm_lengths: Union[np.ndarray, float],
                                               mid_arm_length: float,
                                               long_arm_length: float,
                                               plot_cavity=True, plot_spectrum=True, plot_dependencies=True):
    if plot_cavity:
        cavity.plot()
        plt.show()
    if plot_spectrum:
        cavity.plot_spectrum(width_over_fsr=0.01)
        plt.show()
    if isinstance(short_arm_lengths, (int, float)):
        short_arm_lengths = np.linspace(collimation_point - short_arm_lengths, collimation_point + short_arm_lengths, N)
    NAs, mode_spacing = generate_lens_position_dependencies(short_arm_lengths=short_arm_lengths,
                                                            mid_arm_length=mid_arm_length,
                                                            long_arm_length=long_arm_length,
                                                            plot_dependencies=plot_dependencies)
    mode_spacing_interp = interp1d(mode_spacing, NAs, fill_value='extrapolate')
    mode_spacing_over_fsr_interp = lambda x: mode_spacing_interp(x * cavity.free_spectral_range)
    return mode_spacing_interp, mode_spacing_over_fsr_interp


if __name__ == "__main__":
    mode_spacing_interp, mode_spacing_over_fsr_interp = generate_lens_position_dependencies_output(short_arm_lengths=np.linspace(0.1, 0.9, 10),
                                                                                                   mid_arm_length=0.5,
                                                                                                   long_arm_length=1.0,
                                                                                                   plot_cavity=True,
                                                                                                   plot_spectrum=True,
                                                                                                   plot_dependencies=True
                                                                                                   )
