from cavity import *
np.set_printoptions(precision=10, linewidth=151)


def print_parameters_func(local_parameters):
    for key, value in local_parameters.items():
        # if key.startswith("t_") or key.startswith("p_") or key in ["t", "p"]:
        # print(f"{key} = 1j*{value/np.pi:.10e}")
        if isinstance(value, float):
            print(f"{key} = {value:.10e}")
        elif isinstance(value, str):
            print(f"{key} = '{value}'")
        else:
            print(f"{key} = {value}")


# with open('data/params_dict.pkl', 'rb') as f:
# params_dict = pkl.load(f)

# %matplotlib inline

NA_left = 1.5600000000e-01
x_2_perturbation = 0.0000000000e+00
waist_to_lens = 5.0000000000e-03
waist_to_lens_fine = 0.0000000000e+00
T_edge = 1.0000000000e-03
h = 3.8750000000e-03
set_R_left_to_collimate = True
R_left = 5.4780000000e-03
R_left_fine = 9.8000000000e-06
set_R_right_to_collimate = False
set_R_right_to_equalize_angles = True
set_R_right_to_R_left = False
R_right = 5.4780000000e-03
R_right_fine = 9.8000000000e-06
right_arm_mode = 'fixed length'
right_arm_mode_value = 2.6597084700e-01
lens_fixed_properties = 'sapphire'
mirrors_fixed_properties = 'ULE'
auto_set_x = True
x_span = -1.5700000000e+00
auto_set_y = True
y_span = -2.9000000000e+00
camera_center = 2
add_unheated_cavity = False
print_input_parameters = True
print_cavity_parameters = False

# index 1 represents the right arm, index 2 the inner part of the lens, and index 3 the left arm.
assert not (
            set_R_left_to_collimate and set_R_right_to_collimate), "Too many solutions: can't set automatically both R_left to collimate and R_right to collimate"
assert not (
                       set_R_right_to_collimate + set_R_right_to_equalize_angles + set_R_right_to_R_left) > 1, "Too many constraints on R_right"
if print_input_parameters:
    print_parameters_func(locals())

waist_to_lens += waist_to_lens_fine
R_left += R_left_fine
R_right += R_right_fine
x_span = 10 ** x_span
y_span = 10 ** y_span

cavity = mirror_lens_mirror_cavity_generator(NA_left=NA_left, waist_to_lens=waist_to_lens, h=h, R_left=R_left,
                                             R_right=R_right, T_c=0,
                                             T_edge=T_edge, lens_fixed_properties=lens_fixed_properties,
                                             mirrors_fixed_properties=mirrors_fixed_properties, symmetric_left_arm=True,
                                             waist_to_left_mirror=5e-3, lambda_0_laser=1064e-9, power=2e4,
                                             set_h_instead_of_w=True, right_arm_mode=right_arm_mode,
                                             right_arm_mode_value=right_arm_mode_value,
                                             set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
                                             set_R_right_to_R_left=set_R_right_to_R_left,
                                             set_R_left_to_collimate=set_R_left_to_collimate,
                                             set_R_right_to_collimate=set_R_right_to_collimate)

plot_mirror_lens_mirror_cavity_analysis(cavity,
                                        auto_set_x=auto_set_x,
                                        x_span=x_span,
                                        auto_set_y=auto_set_y,
                                        y_span=y_span,
                                        T_edge=T_edge,
                                        camera_center=camera_center,
                                        add_unheated_cavity=add_unheated_cavity)
plt.show()
if print_cavity_parameters:
    print(cavity.params)