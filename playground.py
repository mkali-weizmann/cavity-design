from matplotlib import use
use("TkAgg")  # or 'Qt5Agg', 'GTK3Agg', etc. depending on your system

from simple_analysis_scripts.potential_analysis.analyze_potential import *


lens_types = ['aspheric - lab', 'spherical - like labs aspheric', 'avantier', 'aspheric - like avantier']
lens_type = lens_types[0]
dn = 0
n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = known_lenses_generator(lens_type=lens_type,
                                                                                                    dn=dn)
desired_focus = 200e-3
cavity = generate_negative_lens_cavity(n_actual_first_lens=n_actual,
                              n_design_first_lens=n_design,
                              T_c_first_lens=T_c,
                              back_focal_length_first_lens=back_focal_length,
                              R_1_first_lens=R_1,
                              R_2_first_lens=R_2,
                              R_2_signed_first_lens=R_2_signed,
                              diameter_first_lens=diameter,
                              approximate_focus_distance_long_arm=desired_focus,
                              negative_lens_focal_length=-15e-3,
                              negative_lens_R_1_inverse=-1/20e-3,
                              negative_lens_back_relative_position=0.1,
                              negative_lens_refractive_index=1.45,
                              negative_lens_center_thickness=4e-3,
                              first_arm_NA=0.15,
                              # right_mirror_ROC=20e-3,
                              right_mirror_distance_to_negative_lens_front=10e-3,
                                       )
incidence_angle_right = calculate_incidence_angle(surface=cavity.surfaces[-2], mode_parameters=cavity.arms[4].mode_parameters)
incidence_angle_left = calculate_incidence_angle(surface=cavity.surfaces[-3], mode_parameters=cavity.arms[2].mode_parameters)
plt.close('all')
ax = cavity.plot()
title = (f"NA short arm = {cavity.arms[0].mode_parameters.NA[0]:.3f}, NA middle arm = {cavity.arms[2].mode_parameters.NA[0]:.3f}, NA right arm = {cavity.arms[4].mode_parameters.NA[0]:.3f}\n"
         f"incidence angle left = {incidence_angle_left:.2f} deg, incidence angle right = {incidence_angle_right:.2f} deg")
ax.set_title(title)
ax.set_xlim(0.323, 0.333)
ax.set_ylim(-0.005, 0.005)
plt.show()