import numpy as np

refractive_indices = np.array([1, 1.5, 1])
Ls = np.array([0.01, 0.005, 0.01])
Rs = np.array([0.01, 0.1])

def snells_law(theta, n1, n2):
    return np.arcsin(n1 * np.sin(theta) / n2)

thetas = np.array([0.01, snells_law(0.01, refractive_indices[0], refractive_indices[1]),
                  -0.03, snells_law(-0.03, refractive_indices[1], refractive_indices[2])])

def ABCD_free_space(L, n):
    return np.array([[1, L / n], [0, 1]])

def ABCD_interface_in_plane(R, theta_1, theta_2, n_1, n_2):
    delta_n_effective = n_2 * np.cos(theta_2) - n_1 * np.cos(theta_1)
    ABCD = np.array([[1, 0], [delta_n_effective / R, 1]])
    return ABCD

def ABCD_interface_out_of_plane(R, theta_1, theta_2, n_1, n_2):
    delta_n_effective = n_2 * np.cos(theta_2) - n_1 * np.cos(theta_1)
    ABCD = np.array([[np.cos(theta_2) / np.cos(theta_1), 0], [delta_n_effective / R, np.cos(theta_1) / np.cos(theta_2)]])
    return ABCD

def reverse_ABCD_matrix(ABCD):
    # I just notice that is True:
    ABCD_reversed = np.array([[ABCD[0, 0]**-1, ABCD[0, 1]], [ABCD[1, 0], ABCD[1, 1] ** -1]])
    return ABCD_reversed

def point_source_output_ROC(ABCD):
    R_output = - ABCD[0, 1] / ABCD[1, 1]
    return R_output


ABCD_1 = ABCD_free_space(Ls[0], refractive_indices[0])
ABCD_2 = ABCD_interface_in_plane(Rs[0], thetas[0], thetas[1], refractive_indices[0], refractive_indices[1])
ABCD_3 = ABCD_free_space(Ls[1], refractive_indices[1])
ABCD_4 = ABCD_interface_in_plane(Rs[1], thetas[2], thetas[3], refractive_indices[1], refractive_indices[2])
ABCD_5 = ABCD_free_space(Ls[2], refractive_indices[2])

ABCD_list = [ABCD_1, ABCD_2, ABCD_3, ABCD_4, ABCD_5]
ABCD_list_reversed = [reverse_ABCD_matrix(ABCD) for ABCD in ABCD_list[::-1]]


ABCD_total = np.linalg.multi_dot(ABCD_list)
ABCD_total_reversed = np.linalg.multi_dot(ABCD_list_reversed)


print(point_source_output_ROC(ABCD_total))
print(point_source_output_ROC(ABCD_total_reversed))