import numpy as np

n_1 = 1
n_2 = 1.8
d_i = 0.004647037041069601
R_1 = 24.22e-3
d_in = 2.91e-3
R_2 = -5.49e-3
d_o = 2e-1


# ABCD_0 = np.array([[1, d_i / n_1], [0, 1]])
# ABCD_1 = np.array([[1, 0], [-(n_2 - n_1)/R_1, 1]])
# ABCD_in = np.array([[1, d_in / n_2], [0, 1]])
# ABCD_2 = np.array([[1, 0], [-(n_1 - n_2)/R_2, 1]])
# ABCD_o = np.array([[1, d_o / n_1], [0, 1]])

ABCD_0 = np.array([[1, d_i], [0, 1]])
ABCD_1 = np.array([[1, 0], [-(n_2 - n_1)/(R_1 * n_2), (n_1/n_2)]])
ABCD_in = np.array([[1, d_in], [0, 1]])
ABCD_2 = np.array([[1, 0], [-(n_1 - n_2)/(R_2 * n_1), (n_2/n_1)]])
ABCD_o = np.array([[1, d_o], [0, 1]])



M_reduced = ABCD_2 @ ABCD_in
M_total = ABCD_o @ ABCD_2 @ ABCD_in @ ABCD_1 @ ABCD_0
M_system = ABCD_2 @ ABCD_in @ ABCD_1
d_o_alleged = -(M_system[0, 0] * d_i + M_system[0, 1]) / (M_system[1, 0] * d_i + M_system[1, 1])

print(f"{M_total[0, 1]=} (should be = 0)")
print(f"{d_o=} (this one is the one that makes M[0, 1] = 0)")
print(f"{d_o_alleged=} (Why is this not the same as d_o?)")


