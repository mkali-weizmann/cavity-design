import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

measured_reflectance = pd.read_csv('data/Sapphire FiveNine reflectance measurement 260323.csv')
measured_reflectance['reflection_calibrated_uW'] = measured_reflectance['reflection_uW'] - measured_reflectance['dark_uW']
measured_reflectance['R'] = measured_reflectance['reflection_calibrated_uW'] / measured_reflectance['input_power_uW']
measured_reflectance.sort_values(['polarization', 'angle_deg'], inplace=True)
s_mask = measured_reflectance['polarization'] == 's'
p_mask = measured_reflectance['polarization'] == 'p'
df_s = measured_reflectance[s_mask]
df_p = measured_reflectance[p_mask]

plt.scatter(df_s['angle_deg'], df_s['R'])
plt.scatter(df_p['angle_deg'], df_p['R'])
plt.xlabel('Angle of Incidence (degrees)')
plt.ylabel('Reflectance')
plt.yscale('log')
plt.title('Measured Reflectance vs Angle of Incidence for Sapphire')
plt.legend(['s-polarization', 'p-polarization'])
plt.grid()
plt.show()
# %%
def find_crossing_linear(x, y, y0):
    diff = y - y0
    idx = np.where(diff[:-1] * diff[1:] <= 0)[0]

    if len(idx) == 0:
        raise ValueError("No crossing found.")
    if len(idx) > 1:
        raise ValueError("More than one crossing found.")

    i = idx[0]
    x1, x2 = x[i], x[i+1]
    y1, y2 = y[i], y[i+1]

    return x1 + (y0 - y1) * (x2 - x1) / (y2 - y1)

dfs_calibrated = []
for df in [df_s, df_p]:
    positive_angles_df = df[df['angle_deg'] > 0]
    minimal_angle = df.iloc[0]['angle_deg']
    reflectance_at_minimal_angle = df.iloc[0]["R"]
    positive_angle_of_same_reflectance = find_crossing_linear(
        positive_angles_df['angle_deg'].to_numpy(),
        positive_angles_df['R'].to_numpy(),
        reflectance_at_minimal_angle
    )
    angle_shift = minimal_angle + positive_angle_of_same_reflectance
    df_calibrated = df.copy()
    df_calibrated['angle_deg'] = df_calibrated['angle_deg'] - angle_shift / 2
    dfs_calibrated.append(df_calibrated)

df_s_calibrated, df_p_calibrated = dfs_calibrated

plt.scatter(np.abs(df_s_calibrated['angle_deg']), df_s_calibrated['R'])
plt.scatter(np.abs(df_p_calibrated['angle_deg']), df_p_calibrated['R'])
plt.xlabel('Angle of Incidence (degrees)')
plt.ylabel('Reflectance')
plt.yscale('log')
plt.title('Measured Reflectance vs Angle of Incidence for Sapphire (Calibrated)')
plt.legend(['s-polarization', 'p-polarization'])
plt.grid()
plt.show()

# %%
def gaussian_smooth(x, y, sigma):
    y_out = np.empty_like(y)

    for i, xi in enumerate(x):
        w = np.exp(-(x - xi)**2 / (2*sigma**2))
        y_out[i] = np.sum(w*y) / np.sum(w)

    return y_out

reflectance_s = gaussian_smooth(np.abs(df_s_calibrated['angle_deg'].to_numpy()), df_s_calibrated['R'].to_numpy(), sigma=1)
reflectance_p = gaussian_smooth(np.abs(df_p_calibrated['angle_deg'].to_numpy()), df_p_calibrated['R'].to_numpy(), sigma=1)
plt.scatter(np.abs(df_s_calibrated['angle_deg']), reflectance_s)
plt.scatter(np.abs(df_p_calibrated['angle_deg']), reflectance_p)
plt.xlabel('Angle of Incidence (degrees)')
plt.ylabel('Reflectance (Smoothed)')
plt.yscale('log')
plt.title('Measured Reflectance vs Angle of Incidence for Sapphire (Calibrated and Smoothed)')
plt.legend(['s-polarization', 'p-polarization'])
plt.grid()
plt.show()
# %%
R_s = scipy.interpolate.interp1d(np.abs(df_s_calibrated['angle_deg']), reflectance_s, kind='linear', fill_value='extrapolate')
R_p = scipy.interpolate.interp1d(np.abs(df_p_calibrated['angle_deg']), reflectance_p, kind='linear', fill_value='extrapolate')
angles = np.linspace(0, 30, 1000)
plt.plot(angles, R_s(angles))
plt.plot(angles, R_p(angles))
plt.xlabel('Angle of Incidence (degrees)')
plt.ylabel('Reflectance (Smoothed and Interpolated)')
plt.yscale('log')
plt.title('Measured Reflectance vs Angle of Incidence for Sapphire (Calibrated, Smoothed, and Interpolated)')
plt.legend(['s-polarization', 'p-polarization'])
plt.grid()
plt.show()

# %%

λ = 1064e-9
P_0 = 2.8e4
Rs = (24.22e-3, 5.49e-3)
NAs = (0.156, 6.6e-3)
ds = (5e-3, 15e-2)
D = 7.75e-3  # Diameter of the lens
i=0
R = Rs[i]
NA = NAs[i]
d = ds[i]
w_0 = λ / (np.pi * NA)
z_R = np.pi * w_0 ** 2 / λ
w = w_0 * np.sqrt(1 + (d / z_R) ** 2)
β = np.linspace(0, np.pi / 6, 1000)
α = np.arcsin(R / (R + d) * np.sin(β))
ρ = d * np.tan(α)
dP_dρ = P_0 * 4 * ρ / w ** 2 * np.exp(-2 * (ρ / w) ** 2)
dρ_dα = d / np.cos(α) ** 2
dα_dβ = R / (R + d) * np.cos(β) / np.sqrt(1 - (R / (R + d) * np.sin(β)) ** 2)
dP_dβ = dP_dρ * dρ_dα * dα_dβ
beta_degrees = β / (2 * np.pi) * 360  # Convert to degrees
dP_dβ_normalized = dP_dβ / np.trapezoid(dP_dβ, beta_degrees)  # Normalize the power over the angle range
dP_dβ_normalized_interpolated = scipy.interpolate.interp1d(beta_degrees, dP_dβ_normalized, kind='linear', fill_value='extrapolate')
plt.figure(figsize=(10, 6))
plt.plot(beta_degrees, dP_dβ_normalized, label='Total Power vs Angle of Incidence')
plt.xlabel('Angle of Incidence (degrees)')
plt.ylabel('Power (Normalized)')
plt.xlim(0, 40)
plt.title(f'Total Power vs Angle of Incidence for a Lens\nR={R * 1e3:.2f} mm')
plt.grid()
plt.legend()
plt.show()

# %%
plt.plot(beta_degrees, dP_dβ_normalized_interpolated(beta_degrees) * R_s(beta_degrees))
plt.plot(beta_degrees, dP_dβ_normalized_interpolated(beta_degrees) * R_p(beta_degrees))
plt.xlabel('Angle of Incidence (degrees)')
plt.ylabel('Power (Normalized and Weighted by Reflectance)')
plt.xlim(0, 40)
plt.title(f'Total Reflected Power vs Angle of Incidence')
plt.grid()
plt.legend(['s-polarization', 'p-polarization'])
plt.show()
# %%
total_reflected_power_s = np.trapezoid(dP_dβ_normalized_interpolated(beta_degrees) * R_s(beta_degrees), beta_degrees)
total_reflected_power_p = np.trapezoid(dP_dβ_normalized_interpolated(beta_degrees) * R_p(beta_degrees), beta_degrees)
print(f'Total Reflected Power for s-polarization: {total_reflected_power_s:.3e}')
print(f'Total Reflected Power for p-polarization: {total_reflected_power_p:.3e}')


