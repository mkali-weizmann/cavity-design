from matplotlib import use
use("TkAgg")  # or 'Qt5Agg', 'GTK3Agg', etc. depending on your system
from simple_analysis_scripts.potential_analysis.analyze_potential import *


# %%
dn = 0
lens_types = ['aspheric - lab', 'spherical - like labs aspheric', 'avantier', 'aspheric - like avantier']
lens_type = lens_types[0]
n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = known_lenses_generator(lens_type=lens_type,
                                                                                                    dn=dn)
n_rays = 400
unconcentricity = 5e-4  # np.float64(0.007610344827586207)  # ,  np.float64(0.007268965517241379)
phi_max = 0.04
desired_focus = 200e-3
plot = True
print_tests = True

defocus = choose_source_position_for_desired_focus_analytic(desired_focus=desired_focus, T_c=T_c, n_design=n_design, diameter=diameter, back_focal_length=back_focal_length, R_1=R_1, R_2=R_2_signed,)
optical_system, optical_axis = generate_one_lens_optical_system(R_1=R_1, R_2=R_2_signed, back_focal_length=back_focal_length, defocus=defocus, T_c=T_c, n_design=n_design, diameter=diameter, n_actual=n_actual,)
rays_0 = initialize_rays(defocus=defocus, n_rays=n_rays, phi_max=phi_max, diameter=diameter, back_focal_length=back_focal_length)
results_dict = analyze_potential(optical_system=optical_system, rays_0=rays_0, unconcentricity=unconcentricity, print_tests=print_tests)

if print_tests:
    print(
        f"Defocus solution for 30 mm focus: {defocus*1e3:.3f} mm, focal point distance: {(results_dict['center_of_curvature'][0] - results_dict['optical_system'].physical_surfaces[1].center[0]) * 1e3:.2f} mm"
    )
    print(
        f"Paraxial spot size for unconcentricity of {unconcentricity*1e6:.1f} μm: {results_dict['spot_size_paraxial']*1e3:.2f} mm"
    )
    print(
        f"Boundary of 2nd vs 4th order dominance for unconcentricity of {unconcentricity*1e6:.1f} µm: {np.abs(results_dict['zero_derivative_points']*1e3):.2f} mm"
    )
if plot:
    # plt.close('all')
    fig, ax = plot_results(results_dict, far_away_plane=True, unconcentricity=unconcentricity, potential_x_axis_angles=False, rays_labels=["Before lens", "After flat surface", "After aspheric surface"])
    center = results_dict["center_of_curvature"]
    ax[1, 0].set_xlim((-0.01, 1))
    plt.suptitle(
        f"lens_type={lens_type}, desired_focus = {desired_focus:.3e}m, n_design: {n_design:.3f}, n_actual: {n_actual:.3f}, Lens focal length: {back_focal_length * 1e3:.1f} mm, Defocus: z_lens -> z_lens + {defocus * 1e3:.1f} mm, T_c: {T_c * 1e3:.1f} mm, Diameter: {diameter * 1e3:.2f} mm"
    )
    ax[1, 1].set_xlim(-0.1, 1)
    # Save image with suptitle in name:
    # plt.savefig(
    #     f"outputs/figures/analyze_potential_n_design lens_type={lens_type}_{n_design:.3f}_n_actual_{n_actual:.3f}_focal_length_{back_focal_length * 1e3:.1f}mm_defocus_{defocus * 1e3:.1f}mm_Tc_{T_c * 1e3:.1f}mm_diameter_{diameter * 1e3:.2f}mm.svg",
    #     dpi=300,
    # )
    plt.show()


# %% FOR-LOOP ANALYSIS:
lens_types = ['aspheric - lab', 'spherical - like labs aspheric', 'avantier', 'aspheric - like avantier']
dn=0
desired_focus=200e-3
n_rays=400
phi_max=0.21
lens_type = lens_types[0]
n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = known_lenses_generator(lens_type, dn)
maximal_unconcentricitiy = 1e-3  # np.float64(0.007610344827586207)  # ,  np.float64(0.007268965517241379)
print_tests=True

defocus = choose_source_position_for_desired_focus_analytic(desired_focus=desired_focus, T_c=T_c, n_design=n_design, diameter=diameter, back_focal_length=back_focal_length, R_1=R_1, R_2=R_2_signed)
unconcentricities = np.linspace(maximal_unconcentricitiy, 0.1e-3, 30)
paraxial_spot_sizes = np.zeros_like(unconcentricities)
spot_size_boundaries = np.zeros_like(unconcentricities)
paraxial_NAs = np.zeros_like(unconcentricities)
left_NAs = np.zeros_like(unconcentricities)
for i, u in enumerate(unconcentricities):
    defocus = choose_source_position_for_desired_focus_analytic(desired_focus=desired_focus, T_c=T_c, n_design=n_design, diameter=diameter, back_focal_length=back_focal_length, R_1=R_1, R_2=R_2_signed,)
    optical_system, optical_axis = generate_one_lens_optical_system(R_1=R_1, R_2=R_2_signed, back_focal_length=back_focal_length, defocus=defocus, T_c=T_c, n_design=n_design, diameter=diameter, n_actual=n_actual, )
    rays_0 = initialize_rays(defocus=defocus, n_rays=n_rays, phi_max=phi_max, diameter=diameter, back_focal_length=back_focal_length)
    results_dict = analyze_potential(optical_system=optical_system, rays_0=rays_0, unconcentricity=u, print_tests=print_tests)
    paraxial_spot_sizes[i] = results_dict["spot_size_paraxial"]
    paraxial_NAs[i] = results_dict["NA_paraxial"]
    left_NAs[i] = results_dict["cavity"].arms[0].mode_parameters.NA[0]
    try:
        spot_size_boundaries[i] = np.abs(results_dict["zero_derivative_points"])
    except TypeError:
        spot_size_boundaries[i] = np.nan  # If zero_derivative_points is None
# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(unconcentricities * 1e6, paraxial_spot_sizes * 1e3, label="Paraxial spot size", color="blue")
ax.scatter(
    unconcentricities * 1e6, spot_size_boundaries * 1e3, label="Boundary of 2nd vs 4th order dominance", color="red"
)
ax.set_xlabel("Unconcentricity (µm)")
ax.set_ylabel("Spot size / boundary (mm)")
ax.set_title(f"paraxial spot size vs. aberrations limit\nLens type={lens_type}, Lens focal length: ")
ax.grid()

# twin y-axis for paraxial NAs
ax2 = ax.twinx()
ax2.plot(unconcentricities * 1e6, paraxial_NAs, label="Right NA", color="orange", linestyle="--")
ax2.plot(unconcentricities * 1e6, left_NAs, label="Left NA", color="green", linestyle="--")
ax2.set_ylabel("Paraxial NA (unitless)")

# combined legend
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2, loc="best")

plt.savefig(
    f"outputs/figures/spot_size_limit lens_type={lens_type}_n_design_{n_design:.3f}_n_actual_{n_actual:.3f}_focal_length_{back_focal_length * 1e3:.1f}mm_defocus_{defocus * 1e3:.1f}mm_Tc_{T_c * 1e3:.1f}mm_diameter_{diameter * 1e3:.2f}mm.svg"
)
plt.show()
