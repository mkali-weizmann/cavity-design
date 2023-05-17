import numpy as np
import matplotlib.pyplot as plt
from cavity import gaussians_overlap_integral, gaussians_overlap_integral_v2, gaussian_integral_2d, gaussian_norm
D = 10
N = 1000
def evaluate_gaussian(x, y, x_0, y_0, w_x, w_y, theta, k_x, k_y):
    x_rot = np.cos(theta) * (x - x_0) + np.sin(theta) * (y - y_0)
    y_rot = - np.sin(theta) * (x - x_0) + np.cos(theta) * (y - y_0)
    return np.exp(- (x_rot ** 2 / w_x ** 2 + y_rot ** 2 / w_y ** 2)) * np.cos(k_x * (x - x_0) + k_y * (y - y_0))

# w_x_1, w_y_1, w_x_2, w_y_2, x_2, y_2, k_x, k_y, theta
def calculate_gaussian_overlap_numeric(w_x_1, w_y_1, w_x_2, w_y_2, x_2, y_2, k_x, k_y, theta):
    x = np.linspace(-D, D, N)
    y = np.linspace(-D, D, N)
    X, Y = np.meshgrid(x, y)
    Z_1 = evaluate_gaussian(X, Y, x_2, y_2, w_x_1, w_y_1, 0, 0, 0)
    Z_2 = evaluate_gaussian(X, Y, 0, 0, w_x_2, w_y_2, theta, k_x, k_y)

    return np.sum(Z_1 * Z_2) / np.sqrt(np.sum(Z_1 ** 2) * np.sum(Z_2 ** 2))


def evaluate_integrand(x, y, w_x_1, w_y_1, w_x_2, w_y_2, x_2, y_2, k_x, k_y, theta):
    a_x = 1 / w_x_1 ** 2 + (np.cos(theta)) ** 2 / w_x_2 ** 2 + (np.sin(theta)) ** 2 / w_y_2 ** 2
    b_x = -2 * x_2 / w_x_1 ** 2
    a_y = 1 / w_y_1 ** 2 + (np.sin(theta)) ** 2 / w_x_2 ** 2 + (np.cos(theta)) ** 2 / w_y_2 ** 2
    b_y = -2 * y_2 / w_y_1 ** 2
    a = (1 / w_x_2 ** 2 - 1 / w_y_2 ** 2) * np.sin(2 * theta)
    c = -x_2 ** 2 / w_x_1 ** 2 - y_2 ** 2 / w_y_1 ** 2
    exponent_term = -a_x * x ** 2 + (b_x + 1j * k_x) * x - a_y * y ** 2 + (b_y + 1j * k_y) * y - a * x * y + c
    expression = np.exp(exponent_term)
    return 1/2 * np.real((expression + np.conj(expression)))


# thetas = [0]  #np.linspace(0, np.pi, 40)
# overlap = np.zeros_like(thetas)
# for i, theta in enumerate(thetas):
w_x_1 = 1
w_y_1 = 1
x_2 = 0
y_2 = 0
w_x_2 = 3
w_y_2 = 1
theta = 0.2
k_x = 0
k_y = 0


x = np.linspace(-D, D, N)
y = np.linspace(-D, D, N)
X, Y = np.meshgrid(x, y)
#
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
sigma_2 = rotation_matrix @ np.diag([w_x_2**2, w_y_2**2]) @ rotation_matrix.T
sigma_2_inverse = np.linalg.inv(sigma_2)
R_2 = np.stack([X, Y], axis=2)
R_2_normed_squared = np.einsum('ijk,kl,ijl->ij', R_2, sigma_2_inverse, R_2)
k_vec = np.array([k_x, k_y])
#
# R_2_normed_squared = (np.cos(theta) ** 2 * w_x_2 ** -2 + np.sin(theta) ** 2 * w_y_2 ** -2) * X ** 2 + \
#                      (w_y_2**-2-w_x_2**-2) * np.sin(2*theta) * X * Y + \
#                      (np.sin(theta)**2*w_x_2**-2+np.cos(theta)**2 * w_y_2**-2) * Y ** 2
#
R_1_normed_squared = (X + x_2) ** 2 / w_x_1 ** 2 + (Y + y_2) ** 2 / w_y_1 ** 2

manual_exponent_1 = np.exp(-R_1_normed_squared)
manual_exponent_2 = np.exp(-R_2_normed_squared) * np.cos(R_2 @ k_vec)

functions_exponent_1 = evaluate_gaussian(X, Y, -x_2, -y_2, w_x_1, w_y_1, 0, 0, 0)
functions_exponent_2 = evaluate_gaussian(X, Y, 0, 0, w_x_2, w_y_2, theta, k_x, k_y)

manual_integrand = manual_exponent_1 * manual_exponent_2
functions_manual_integrand = functions_exponent_1 * functions_exponent_2
functions_integrand = evaluate_integrand(X, Y, w_x_1, w_y_1, w_x_2, w_y_2, x_2, y_2, k_x, k_y, theta)

plt.imshow(manual_exponent_2)
plt.colorbar()
plt.show()
#
# plt.imshow(functions_manual_integrand)
# plt.colorbar()
# plt.show()
#
# plt.imshow(functions_integrand)
# plt.colorbar()
# plt.show()

# i, j = 60, 50
# print([manual_integrand[i, j], functions_integrand[i, j]])
# Z = np.exp(-R_normed_with_sigma_2_inv)

I_manual = np.sum(manual_integrand) / np.sqrt(np.sum(manual_exponent_1**2) * np.sum(manual_exponent_2**2))
I_numeric = calculate_gaussian_overlap_numeric(w_x_1, w_y_1, w_x_2, w_y_2, x_2, y_2, k_x, k_y, theta)
I_functions_integrand = np.sum(functions_manual_integrand) / np.sqrt(np.sum(functions_exponent_1**2) * np.sum(functions_exponent_2**2))
I_analytic = gaussians_overlap_integral(w_x_1, w_y_1, w_x_2, w_y_2, x_2, y_2, k_x, k_y, theta)
I_analytic_v2 = gaussians_overlap_integral_v2(w_x_1, w_y_1, w_x_2, w_y_2, x_2, y_2, k_x, k_y, theta)

print(f"I_manual                   = {I_manual}")
print(f"I_numeric                  = {I_numeric}")
print(f"I_functions_integrand      = {I_functions_integrand}")
print(f"I_analytic                 = {I_analytic}")
print(f"I_analytic_v2              = {I_analytic_v2}")
print(f"I_analytic_v2  / I_numeric = {I_analytic_v2  / I_numeric}")

# overlap[i] = I_analytic_v2  / I_numeric

# plt.plot(thetas, overlap)
# plt.show()
a_x = 2 / w_x_1 ** 2
a_y = 2 / w_y_1 ** 2
print(gaussian_integral_2d(a_x, 0, 0, a_y, 0, 0, 0, 0))
print(gaussian_norm(w_x_1, w_y_1, 0, 0, 0)**2)
print(np.sum(manual_exponent_1**2) * (x[1] - x[0]) * (y[1] - y[0]))

print(" ")
a_x = 2 * (np.cos(theta) ** 2 / w_x_2 ** 2 + np.sin(theta) ** 2 / w_y_2 ** 2)
a_y = 2 * (np.sin(theta) ** 2 / w_x_2 ** 2 + np.cos(theta) ** 2 / w_y_2 ** 2)
a = 2 * np.sin(2 * theta) * (1 / w_x_2 ** 2 - 1 / w_y_2 ** 2)
# %%
kaki = evaluate_integrand(X, Y, 50, 50, w_x_2, w_y_2, 0, 0, 0, 0, 0.2)
plt.imshow(kaki)
plt.colorbar()
plt.show()


