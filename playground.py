# %% Imports and constants
import numpy as np
beta = 1e-6
alpha_m = 7.5e-8
alpha_l = 5.5e-6
alpha_vol = 0.01
dn_dT = 1.17e-5
w_m_1 = 0.0024779653724884094
w_l_1 = w_m_1
w_m_2 = 0.002
w_l_2 = w_m_2

P = 2e4
k_m = 1.31
k_l = 46.06
nu_m = 0.17
nu_l = 0.3
c_nu_m = (1+nu_m) / (1-nu_m)
c_nu_l = (1+nu_l) / (1-nu_l)
w_half_lens = 5.9e-3 * 0.5
R_lens = 0.015166312489
n_lens = 1.76

# %% First mirror's 1/f:
delta_T_m_1 = beta * P / (k_m * w_m_1)
delta_z_m_1 = alpha_m * c_nu_m * beta * P / k_m
delta_curvature_m_1 = delta_z_m_1 / w_m_1**2
delta_f_m_1_inv = 2 * delta_curvature_m_1

# %% Second mirror's 1/f
delta_T_m_2 = beta * P / (k_m * w_m_2)
delta_z_m_2 = alpha_m * c_nu_m * beta * P / k_m
delta_curvature_m_2 = delta_z_m_2 / w_m_2**2
delta_f_m_2_inv = 2 * delta_curvature_m_2

# %% First lens side 1/f:
delta_T_vol_l = alpha_vol * (P / k_l)
delta_T_surface_l_1 = beta * P / (k_l * w_l_1)
dxi_dy2_n_surface = (P / k_l) * beta * dn_dT * w_l_1 ** -2
dxi_dy2_n_vol_1_radius_term = (P / k_l) * alpha_vol * dn_dT / R_lens
dxi_dy2_n_vol_1_profile_term = (P / k_l) * alpha_vol * dn_dT * (w_half_lens / w_l_1 ** 2)
dxi_dy2_curvature = (P / k_l) * n_lens * alpha_l * beta * c_nu_l * w_l_1 ** -2

delta_n_surface_term = - dxi_dy2_n_surface * R_lens
delta_n_radius_term = - dxi_dy2_n_vol_1_radius_term * R_lens
delta_n_radius_vol_profile_term = - dxi_dy2_n_vol_1_profile_term * R_lens
delta_n_curvature_term = - dxi_dy2_curvature * R_lens


def f_inverse(n):
    return (n-1) * (2/R_lens - (n-1) * 2*w_half_lens / (n * R_lens**2))

delta_f_surface_term = f_inverse(n_lens + delta_n_surface_term) - f_inverse(n_lens)
delta_f_radius_term = f_inverse(n_lens + delta_n_radius_term) - f_inverse(n_lens)
delta_f_radius_vol_profile_term = f_inverse(n_lens + delta_n_radius_vol_profile_term) - f_inverse(n_lens)
delta_f_curvature_term = f_inverse(n_lens + delta_n_curvature_term) - f_inverse(n_lens)