import numpy as np
lambda_0_laser = 1064e-9
n_1 = 1
n_2 = 1.76
R_1 = 0.010596029  # 0.023479617901  #
R_2 = -0.005  # -0.0062124542986  #
d = 0.0035741639440999987  # 0.0026786030286000016  #

def M(q, A, B, C, D):
    return (A * q + B) / (C * q + D)

def w(q, n):
    return np.sqrt(-lambda_0_laser / n ** 2 / (np.pi * np.imag(1 / q)))

def w_alternative(q, n):
    return np.sqrt(lambda_0_laser / n**2 * q.imag / np.pi) * np.sqrt(1 + (q.real / q.imag)**2 )

def w_hat(q_hat, n):
    return np.sqrt(-lambda_0_laser / n / (np.pi * np.imag(1 / q_hat)))

def w_hat_alternative(q_hat, n):
    return np.sqrt(lambda_0_laser / n * q_hat.imag / np.pi) * np.sqrt(1 + (q_hat.real / q_hat.imag)**2 )

q_1 = 0.00500255+0.00011355j  # 0.005 + 1.77841689e-05j  #
w_1 = w(q_1, n_1)
w_1_alternative = w_alternative(q_1, n_1)

# Calculate \hat{q}_{2} using the given equation
q_2_hat = M(q_1, 1, 0, (n_1 - n_2) / (R_1 * n_2), n_1 / n_2)
q_2_hat_times_n_2 = q_2_hat * n_2
q_2 = n_2 * M(q_1 / n_1, 1, 0, (n_1 - n_2) / (R_1 * n_2), n_1 / n_2)
w_2 = w(q_2, n_2)
w_2_alternative = w_alternative(q_2, n_2)
w_2_hat = w_hat(q_2_hat, n_2)
w_2_hat_alternative = w_hat_alternative(q_2_hat, n_2)

q_3_hat = M(q_2_hat, 1, d, 0, 1)
q_3_hat_times_n = q_3_hat * n_2
q_3 = n_2 * M(q_2 / n_2, 1, d, 0, 1)
w_3 = w(q_3, n_2)
w_3_alternative = w_alternative(q_3, n_2)
w_3_hat = w_hat(q_3_hat, n_2)
w_3_hat_alternative = w_hat_alternative(q_3_hat, n_2)

q_4_hat = M(q_3_hat, 1, 0, (n_2 - n_1) / (R_2 * n_1), n_2 / n_1)
q_4_hat_times_n = q_4_hat * n_1
q_4 = n_1 * M(q_3 / n_2, 1, 0, (n_2 - n_1) / (R_2 * n_1), n_2 / n_1)
w_4 = w(q_4, n_1)
w_4_alternative = w_alternative(q_4, n_1)