from matplotlib import use
use("TkAgg")  # or 'Qt5Agg', 'GTK3Agg', etc. depending on your system
import sys
from pathlib import Path

project_root = Path.cwd().parent.parent
sys.path.append(str(project_root))

from ipywidgets import Layout, FloatSlider, Checkbox, Text, widgets, IntSlider, Dropdown

from simple_analysis_scripts.potential_analysis.analyze_potential import *

import io
import matplotlib.pyplot as plt
from PIL import Image
# import win32clipboard

# def copy_figure_to_clipboard(fig):
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
#     buf.seek(0)
#
#     image = Image.open(buf)
#     output = io.BytesIO()
#     image.convert("RGB").save(output, "BMP")
#     data = output.getvalue()[14:]  # BMP header removed
#
#     win32clipboard.OpenClipboard()
#     win32clipboard.EmptyClipboard()
#     win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
#     win32clipboard.CloseClipboard()

from simple_analysis_scripts.potential_analysis.analyze_potential import *
# %%
OPTICAL_AXIS = RIGHT
copy_image = False
copy_input_parameters = True
copy_cavity_parameters = False
lens_type = 'aspheric - lab'
desired_focus = 2.0000000000e-01
n_rays = 50
phi_max = 1.5000000000e-01
n_actual_spherical = 1.4500000000e+00
relative_position_spherical = 3.0000000000e-01
negative_lens_focal_length = -2.2400000000e-02
negative_lens_R_1_inverse = -5.0000000000e+01
negative_lens_back_relative_position = 1.0000000000e-01
negative_lens_refractive_index = 1.4500000000e+00
negative_lens_center_thickness = 4.0000000000e-03
mirror_setting_mode = 'Set distance to spherical'
first_arm_NA = 1.5000000000e-01
right_mirror_ROC = 2.0000000000e-01
right_mirror_distance_to_negative_lens_front = 1.0000000000e-02
right_mirror_ROC_fine = 9.5393584798e-11

# def f(
#         copy_image, copy_input_parameters, copy_cavity_parameters, lens_type, desired_focus, n_rays, phi_max, n_actual_spherical, relative_position_spherical, negative_lens_focal_length, negative_lens_R_1_inverse, negative_lens_back_relative_position, negative_lens_refractive_index, negative_lens_center_thickness, mirror_setting_mode, first_arm_NA, right_mirror_ROC, right_mirror_distance_to_negative_lens_front, right_mirror_ROC_fine) :
right_mirror_ROC_fine = widget_convenient_exponent(right_mirror_ROC_fine, scale=-10)
if copy_input_parameters:
    copy_parameters_func(locals())
if mirror_setting_mode == "Set ROC":
    right_mirror_distance_to_negative_lens_front = None
    right_mirror_ROC += right_mirror_ROC_fine
elif mirror_setting_mode == "Set distance to spherical":
    right_mirror_ROC = None
    right_mirror_distance_to_negative_lens_front += right_mirror_ROC_fine
n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = known_lenses_generator(
    lens_type=lens_type,
    dn=0)
cavity = generate_negative_lens_cavity(n_actual_first_lens=n_actual, n_design_first_lens=n_design, T_c_first_lens=T_c, back_focal_length_first_lens=back_focal_length, R_1_first_lens=R_1, R_2_first_lens=R_2, R_2_signed_first_lens=R_2_signed, diameter_first_lens=diameter, approximate_focus_distance_long_arm=desired_focus, negative_lens_focal_length=negative_lens_focal_length, negative_lens_R_1_inverse=negative_lens_R_1_inverse, negative_lens_back_relative_position=negative_lens_back_relative_position, negative_lens_refractive_index=negative_lens_refractive_index, negative_lens_center_thickness=negative_lens_center_thickness, first_arm_NA=first_arm_NA, right_mirror_ROC=right_mirror_ROC, right_mirror_distance_to_negative_lens_front=right_mirror_distance_to_negative_lens_front, )
incidence_angle_right = calculate_incidence_angle(surface=cavity.surfaces[-2], mode_parameters=cavity.arms[4].mode_parameters)
incidence_angle_left = calculate_incidence_angle(surface=cavity.surfaces[-3], mode_parameters=cavity.arms[2].mode_parameters)
title = (
    f"NA short arm = {cavity.arms[0].mode_parameters.NA[0]:.3f}, NA middle arm = {cavity.arms[2].mode_parameters.NA[0]:.3f}, NA right arm = {cavity.arms[4].mode_parameters.NA[0]:.3f}\n"
    f"incidence angle left = {incidence_angle_left:.2f} deg, incidence angle right = {incidence_angle_right:.2f} deg")
results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=30, phi_max=0.14, print_tests=False)
fig, ax = plot_results(results_dict=results_dict, far_away_plane=True)
ax[1, 0].set_title(title)
ax[0, 0].set_title(f"right mirror spot size = {cavity.arms[4].mode_parameters_on_surface_1.spot_size[0]:.2e}")

# if copy_image:
#     copy_figure_to_clipboard(fig)

if copy_cavity_parameters:
    pyperclip.copy(results_dict['cavity'].formatted_textual_params)
lens_focal_length = focal_length_of_lens(R_1=-results_dict['cavity'].surfaces[3].radius, R_2=results_dict['cavity'].surfaces[4].radius, n=results_dict['cavity'].surfaces[3].n_2, T_c=np.linalg.norm(results_dict['cavity'].surfaces[4].center - results_dict['cavity'].surfaces[3].center))
mirror_radius = results_dict['cavity'].surfaces[-1].radius
plt.show()


# n_rays = 50
# phi_max = 0.15
# back_focal_length_aspheric = 20e-3
# defocus = -0.002
# back_center = (back_focal_length_aspheric - defocus) * OPTICAL_AXIS
# lens_type = 'aspheric - lab'
# desired_focus = 200e-3
# n_actual_spherical = 1.45
# relative_position_spherical = 0.3
# negative_lens_focal_length = -15e-3
# negative_lens_R_1_inverse = -1 / 20e-3
# negative_lens_back_relative_position = 0.1
# negative_lens_refractive_index = 1.45
# negative_lens_center_thickness = 4e-3
# first_arm_NA = 0.15
# fix_right_mirror_ROC = False
# right_mirror_ROC = 20e-3
# right_mirror_distance_to_negative_lens_front = 10e-3
# right_mirror_ROC_fine = 0
# mirror_setting_mode = "Set ROC"
# widgets.interact(
#     f,
#     lens_type=Dropdown(options=[("Existing Aspheric", 'aspheric - lab'), ('Avantier', 'avantier'), ('Spherical - f, n like labs aspheric', 'spherical - like labs aspheric'), ('Aspherical - f, n like Avantier', 'aspheric - like avantier')], value=lens_type, description="Lens type", style={'description_width': 'initial'}),
#     desired_focus=FloatSlider(value=desired_focus, min=1e-3, max=1.0, step=1e-4, description='desired focus (m)', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
#     n_rays=IntSlider(value=n_rays, min=2, max=2000, step=1, description='n rays', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
#     phi_max=FloatSlider(value=phi_max, min=0.01, max=1.0, step=0.001, description='phi max', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
#     n_actual_spherical=FloatSlider(value=n_actual_spherical, min=1.0, max=2.0, step=0.0001, description='n actual spherical', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
#     relative_position_spherical=FloatSlider(value=relative_position_spherical, min=-1.0, max=3.0, step=0.001, description='relative position spherical', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
#     negative_lens_focal_length=FloatSlider(value=negative_lens_focal_length, min=-0.2, max=0.0, step=1e-4, description='negative lens focal length (m)', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
#     negative_lens_R_1_inverse=FloatSlider(value=negative_lens_R_1_inverse, min=-200.0, max=200.0, step=0.1, description='negative lens R1 inverse (1/m)', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
#     negative_lens_back_relative_position=FloatSlider(value=negative_lens_back_relative_position, min=-1.0, max=2.0, step=0.001, description='negative lens back relative position', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
#     negative_lens_refractive_index=FloatSlider(value=negative_lens_refractive_index, min=1.0, max=2.0, step=0.0001, description='negative lens refractive index', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
#     negative_lens_center_thickness=FloatSlider(value=negative_lens_center_thickness, min=0.1e-3, max=20e-3, step=1e-5, description='negative lens center thickness (m)', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
#     mirror_setting_mode=Dropdown(options=[("Set ROC"), ("Set distance to spherical"), ("Set both")], value=mirror_setting_mode, description="Mirror setting mode", style={'description_width': 'initial'}),
#     first_arm_NA=FloatSlider(value=first_arm_NA, min=0.0, max=1.0, step=0.001, description='first arm NA', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
#     right_mirror_ROC=FloatSlider(value=right_mirror_ROC, min=5e-3, max=0.5, step=1e-4, description='right mirror ROC (m)', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
#     right_mirror_distance_to_negative_lens_front=FloatSlider(value=right_mirror_distance_to_negative_lens_front, min=0.0, max=0.5, step=1e-4, description='right mirror distance to negative lens front (m)', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
#     right_mirror_ROC_fine=FloatSlider(value=right_mirror_ROC, min=-10, max=10, step=1e-4, description='Right ROC/distance - fine', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
#     copy_input_parameters=Checkbox(value=False, description='Copy input parameters', style={'description_width': 'initial'}),
#     copy_cavity_parameters=Checkbox(value=False, description='Copy cavity parameters', style={'description_width': 'initial'}),
#     copy_image=Checkbox(value=False, description='Copy image', style={'description_width': 'initial'}),);