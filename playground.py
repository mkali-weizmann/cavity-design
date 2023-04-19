import numpy as np
import matplotlib.pyplot as plt
from cavity import gaussians_overlap_integral


def evaluate_gaussian(x, y, x_0, y_0, w_x, w_y, theta, k_x, k_y):
    x_rot = np.cos(theta) * (x - x_0) + np.sin(theta) * (y - y_0)
    y_rot = -np.sin(theta) * (x - x_0) + np.cos(theta) * (y - y_0)
    return np.exp(- (x_rot ** 2 / w_x ** 2 + y_rot ** 2 / w_y ** 2)) * np.cos(k_x * x_rot + k_y * y_rot)

# w_x_1, w_y_1, w_x_2, w_y_2, x_2, y_2, k_x, k_y, theta
def calculate_gaussian_overlap_numeric(w_1_x, w_1_y, w_2_x, w_2_y, x_2, y_2, k_x, k_y, theta):
    d = 10

    x = np.linspace(-d, d, 1000)
    y = np.linspace(-d, d, 1000)
    X, Y = np.meshgrid(x, y)
    Z_1 = evaluate_gaussian(X, Y, 0, 0, w_1_x, w_1_y, 0, 0, 0)
    Z_2 = evaluate_gaussian(X, Y, x_2, y_2, w_2_x, w_2_y, theta, k_x, k_y)

    return np.sum(Z_1 * Z_2) / np.sqrt(np.sum(Z_1 ** 2) * np.sum(Z_2 ** 2))


def evaluate_integrand(x, y, w_1_x, w_1_y, w_2_x, w_2_y, x_2, y_2, k_x, k_y, theta):
    a_x = 1 / w_1_x**2 + (np.cos(theta))**2 / w_2_x**2 + (np.sin(theta))**2 / w_2_y**2
    b_x = 2 * x_2 / w_1_x**2 + 1j * k_x
    a_y = 1 / w_1_y**2 + (np.sin(theta))**2 / w_2_x**2 + (np.cos(theta))**2 / w_2_y**2
    b_y = 2 * y_2 / w_1_y**2 + 1j * k_y
    a = (1 / w_2_y**2 - 1 / w_2_x**2) * np.sin(2 * theta)
    c = -x_2**2 / w_1_x**2 - y_2**2 / w_1_y**2
    exponent_term = -a_x * x**2 + b_x * x - a_y * y**2 + b_y * y + a * x * y + c
    expression = np.exp(exponent_term)
    return 1/2 * np.real((expression + np.conj(expression)))

w_1_x = 2
w_1_y = 1
x_2 = 1
y_2 = 2
w_2_x = 2
w_2_y = 3
theta = 0.2
k_x = 0
k_y = 0


x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
rotation_matrix_inv = np.linalg.inv(rotation_matrix)
sigma_2 = rotation_matrix_inv @ np.diag([w_2_x**2, w_2_y**2]) @ rotation_matrix_inv.T
sigma_2_inverse = np.linalg.inv(sigma_2)
R_2 = np.stack([X, Y], axis=2)
R_2_normed_squared = np.einsum('ijk,kl,ijl->ij', R_2, sigma_2_inverse, R_2)
k_vec = np.array([k_x, k_y])

# R_2_normed_squared = (np.cos(theta) ** 2 * w_2_x ** -2 + np.sin(theta) ** 2 * w_2_y ** -2) * X ** 2 + \
#                      (w_2_y**-2-w_2_x**-2) * np.sin(2*theta) * X * Y + \
#                      (np.sin(theta)**2*w_2_x**-2+np.cos(theta)**2 * w_2_y**-2) * Y ** 2

R_1_normed_squared = (X + x_2) ** 2 / w_1_x ** 2 + (Y + y_2) ** 2 / w_1_y ** 2

manual_exponent_1 = np.exp(-R_1_normed_squared)
manual_exponent_2 = np.exp(-R_2_normed_squared) * np.cos(R_2 @ k_vec)

functions_exponent_1 = evaluate_gaussian(X, Y, -x_2, -y_2, w_1_x, w_1_y, 0, 0, 0)
functions_exponent_2 = evaluate_gaussian(X, Y, 0, 0, w_2_x, w_2_y, theta, k_x, k_y)

manual_integrand = manual_exponent_1 * manual_exponent_2
functions_integrand = evaluate_integrand(X, Y, w_1_x, w_1_y, w_2_x, w_2_y, x_2, y_2, k_x, k_y, theta)

plt.imshow(manual_exponent_2)
plt.colorbar()
plt.show()

plt.imshow(functions_exponent_2)
plt.colorbar()
plt.show()

# i, j = 60, 50
# print([manual_integrand[i, j], functions_integrand[i, j]])
# Z = np.exp(-R_normed_with_sigma_2_inv)

I_numeric = calculate_gaussian_overlap_numeric(w_1_x, w_1_y, w_2_x, w_2_y, x_2, y_2, k_x, k_y, theta)
I_analytic = gaussians_overlap_integral(w_1_x, w_1_y, w_2_x, w_2_y, x_2, y_2, k_x, k_y, theta)

print(I_analytic / I_numeric)
print((I_analytic / I_numeric)**2)
print((I_analytic / I_numeric)**(1/2))