from cavity import *
D = 10
N = 1000


def evaluate_gaussian(x, y, x_0, y_0, w_x, w_y, theta, k_x, k_y):
    x_rot = np.cos(theta) * (x - x_0) + np.sin(theta) * (y - y_0)
    y_rot = - np.sin(theta) * (x - x_0) + np.cos(theta) * (y - y_0)
    return np.exp(- (x_rot ** 2 / w_x ** 2 + y_rot ** 2 / w_y ** 2)) * np.cos(k_x * (x - x_0) + k_y * (y - y_0))


def evaluate_gaussian_matrices(X, Y, A, b, c):
    r = np.stack([X, Y], axis=2)
    r_normed_squared = np.einsum('ijk,kl,ijl->ij', r, A, r)
    br = np.einsum('k,ijk->ij', b, r)
    return np.exp(-(1/2) * r_normed_squared + br + c)


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


w_x_1 = 1
w_y_1 = 1 + 0.2j
x_2 = 2
y_2 = 0
w_x_2 = 1 + 0.1j
w_y_2 = 1
theta = 0.2
b_x_2 = 0.2 + 3j
b_y_2 = 0.1


x = np.linspace(-D, D, N)
y = np.linspace(-D, D, N)
X, Y = np.meshgrid(x, y)

sigma_1 = np.diag([(1/2)*w_x_1**2, (1/2)*w_y_1**2])
sigma_1_inverse = np.linalg.inv(sigma_1)

rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
sigma_2 = rotation_matrix @ np.diag([(1/2)*w_x_2**2, (1/2)*w_y_2**2]) @ rotation_matrix.T
sigma_2_inverse = np.linalg.inv(sigma_2)

A_1 = sigma_1_inverse
b_1 = np.array([0, 0])
c_1 = 0

A_2 = sigma_2_inverse
b_2 = np.array([x_2, y_2]) @ A_2 + np.array([b_x_2, b_y_2])
c_2 = -1 / 2 * np.array([x_2, y_2]) @ A_2 @ np.array([x_2, y_2])

R = np.stack([X, Y], axis=2)
mu = np.array([x_2, y_2])
R_2 = R - mu[None, None, :]
R_1_normed_squared = np.einsum('ijk,kl,ijl->ij', R, A_1, R)
R_2_normed_squared = np.einsum('ijk,kl,ijl->ij', R_2, A_2, R_2)


manual_exponent_1 = np.exp(-(1/2)*R_1_normed_squared)
manual_exponent_2 = np.exp(-(1/2)*R_2_normed_squared + np.einsum('k,ijk->ij', np.array([b_x_2, b_y_2]), R))

manual_exponent_1_matrices = evaluate_gaussian_matrices(X, Y, A_1, b_1, c_1)
manual_exponent_2_matrices = evaluate_gaussian_matrices(X, Y, A_2, b_2, c_2)

manual_integrand = np.conjugate(manual_exponent_1) * manual_exponent_2 # functions_exponent_1 = evaluate_gaussian(X, Y, -x_2, -y_2, w_x_1, w_y_1, 0, 0, 0) # functions_exponent_2 = evaluate_gaussian(X, Y, 0, 0, w_x_2, w_y_2, theta, k_x, k_y) # functions_manual_integrand = functions_exponent_1 * functions_exponent_2 # functions_integrand = evaluate_integrand(X, Y, w_x_1, w_y_1, w_x_2, w_y_2, x_2, y_2, k_x, k_y, theta)
manual_integrand_matrices = np.conjugate(manual_exponent_1_matrices) * manual_exponent_2_matrices

plt.imshow(np.real(manual_exponent_2))
plt.colorbar()
plt.show()


I_manual = np.sum(manual_integrand) / np.sqrt(np.sum(np.abs(manual_exponent_1)**2) * np.sum(np.abs(manual_exponent_2)**2))  # I_numeric = calculate_gaussian_overlap_numeric(w_x_1, w_y_1, w_x_2, w_y_2, x_2, y_2, k_x, k_y, theta) # I_functions_integrand = np.sum(functions_manual_integrand) / np.sqrt(np.sum(functions_exponent_1**2) * np.sum(functions_exponent_2**2))  # I_analytic = gaussians_overlap_integral_v2(w_x_1, w_y_1, w_x_2, w_y_2, x_2, y_2, k_x, k_y, theta)  # print(f"I_analytic              = {I_analytic}")
I_manual_matrices = np.sum(manual_integrand_matrices) / np.sqrt(np.sum(np.abs(manual_exponent_1_matrices)**2) * np.sum(np.abs(manual_exponent_2_matrices)**2))
I_analytic_matrices = gaussians_overlap_integral(A_1, A_2, b_1, b_2, c_1, c_2)

# print(f"I_manual                          = {I_manual}")  # print(f"I_numeric               = {I_numeric}") # print(f"I_functions_integrand   = {I_functions_integrand}")
# print(f"I_manual_matrices                 = {I_manual_matrices}")
# print(f"I_analytic_matrices               = {I_analytic_matrices}")
print(f"\nI_analytic_matrices / I_numeric = {I_analytic_matrices / I_manual}\n")


N_1_manual = np.sum(np.abs(manual_exponent_1)**2) * (x[1] - x[0]) * (y[1] - y[0])  # print(f"N_1_gaussian_integral_2d = {N_1_gaussian_integral_2d}")  # N_1_gaussian_integral_2d = gaussian_norm(w_x_1, w_y_1, 0, 0, 0)**2
N_1_matrices = np.exp(gaussian_norm_log(A_1, b_1, c_1) ** 2)
# print(f"N_1_manual               = {N_1_manual}")
# print(f"N_1_matrices             = {N_1_matrices}\n")
print(f"N_1_matrices / N_1_manual  = {N_1_matrices / N_1_manual}\n")

N_2_manual = np.sum(np.abs(manual_exponent_2)**2) * (x[1] - x[0]) * (y[1] - y[0]) # N_2_gaussian_integral_2d = gaussian_norm(w_x_2, w_y_2, k_x, k_y, theta)**2 # print(f"N_2_gaussian_integral_2d = {N_2_gaussian_integral_2d}")
N_2_matrices = np.exp(gaussian_norm_log(A_2, b_2, c_2) ** 2)
# print(f"N_2_manual               = {N_2_manual}")
# print(f"N_2_matrices             = {N_2_matrices}\n")
print(f"N_2_matrices / N_2_manual  = {N_2_matrices / N_2_manual}\n")

exponent_1_integral_manual = np.sum(manual_exponent_1) * (x[1] - x[0]) * (y[1] - y[0])
exponent_1_integral_matrices = np.exp(gaussian_integral_2d_log(A_1, b_1, c_1))
# print(f"exponent_1_integral_manual  = {exponent_1_integral_manual}")
# print(f"exponent_1_integral_matrices= {exponent_1_integral_matrices}\n")
print(f"exponent_1_integral_matrices / exponent_1_integral_manual = {exponent_1_integral_matrices / exponent_1_integral_manual}\n")

a_x = 2 * (np.cos(theta) ** 2 / w_x_2 ** 2 + np.sin(theta) ** 2 / w_y_2 ** 2)
a_y = 2 * (np.sin(theta) ** 2 / w_x_2 ** 2 + np.cos(theta) ** 2 / w_y_2 ** 2)
a = 2 * np.sin(2 * theta) * (1 / w_x_2 ** 2 - 1 / w_y_2 ** 2)
exponent_2_integral_manual = np.sum(manual_exponent_2) * (x[1] - x[0]) * (y[1] - y[0])
exponent_2_integral_manual_matrices = np.sum(manual_exponent_2_matrices) * (x[1] - x[0]) * (y[1] - y[0])
exponent_2_integral_matrices = np.exp(gaussian_integral_2d_log(A_2, b_2, c_2))
# print(f"exponent_2_integral_manual  = {exponent_2_integral_manual}")
# print(f"exponent_2_integral_manual_matrices  = {exponent_2_integral_manual}")
# print(f"exponent_2_integral_matrices= {exponent_2_integral_matrices}\n")
print(f"exponent_2_integral_matrices / exponent_2_integral_manual = {exponent_2_integral_matrices / exponent_2_integral_manual}\n")




