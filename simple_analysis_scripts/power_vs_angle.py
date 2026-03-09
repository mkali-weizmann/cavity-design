import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import use
import scipy
# use('TkAgg')  # Use a non-interactive backend for saving figures
# %% NOTE:
# All expressions here are the results of my derivation at the https://mynotebook.labarchives.com/MTM3NjE3My41fDEwNTg1OTUvMTA1ODU5NS9Ob3RlYm9vay8zMjQzMzA0MzY1fDM0OTMzNjMuNQ==/page/11290221-151
# Or at my research the lyx file under subsection "Power vs angle and radius"
# %%

λ = 1064e-9
P_0 = 2.8e4
Rs = (24.22e-3, 5.49e-3)
NAs = (0.156, 6.6e-3)
ds = (5e-3, 15e-2)
D = 7.75e-3  # Diameter of the lens

for i in range(1, 2):
    R = Rs[i]
    NA = NAs[i]
    d = ds[i]
    w_0 = λ / (np.pi * NA)
    z_R = np.pi * w_0 ** 2 / λ
    w = w_0 * np.sqrt(1 + (d / z_R) ** 2)
    β = np.linspace(0, np.pi / 5, 10000)
    α = np.arcsin(R / (R + d) * np.sin(β))
    ρ = d * np.tan(α)
    dP_dρ = P_0 * 4 * ρ / w ** 2 * np.exp(-2 * (ρ / w) ** 2)
    dρ_dα = d / np.cos(α) ** 2
    dα_dβ = R / (R + d) * np.cos(β) / np.sqrt(1 - (R / (R + d) * np.sin(β)) ** 2)
    dP_dβ = dP_dρ * dρ_dα * dα_dβ
    beta_degrees = β / (2 * np.pi) * 360  # Convert to degrees
    dP_dβ_normalized = dP_dβ / np.trapezoid(dP_dβ, beta_degrees)  # Normalize the power over the angle range
    plt.figure(figsize=(10, 6))
    plt.plot(beta_degrees, dP_dβ_normalized, label='Total Power vs Angle of Incidence')
    plt.xlabel('Angle of Incidence (degrees)')
    plt.ylabel('Power (Normalized)')
    plt.xlim(0, 40)
    plt.title(f'Total Power vs Angle of Incidence for a Lens\nR={R * 1e3:.2f} mm')
    plt.grid()
    plt.legend()
    # plt.savefig(f'figures/total_power_vs_angle_{R * 1e5:.0f}.svg', bbox_inches='tight')
    # plt.show()

    # save dP_dβ_normalized and beta_angles to a csv file:
    # import pandas as pd
    # df = pd.DataFrame({'Angle of Incidence (degrees)': beta_angles, 'Total Power (W)': dP_dβ_normalized})
    # df.to_csv(f'figures/total_power_vs_angle_{R * 1e5:.0f}.csv', index=False)
    # print(f'Saved data for R={R * 1e3:.2f} mm to figures/total_power_vs_angle_{R * 1e5:.0f}.csv')

# %% Power vs radius
for i in range(0, 2):
    R = Rs[i]
    NA = NAs[i]
    d = ds[i]
    w_0 = λ / (np.pi * NA)
    z_R = np.pi * w_0 ** 2 / λ
    w = w_0 * np.sqrt(1 + (d / z_R) ** 2)
    r = np.linspace(0, D / 2, 10000)
    θ = np.arcsin(r / R)
    α = np.arctan(r / (d + R - np.sqrt(R ** 2 - r ** 2)))
    β = θ + α
    β_degrees = β / (2 * np.pi) * 360  # Convert to degrees
    θ_degrees = θ / (2 * np.pi) * 360  # Convert to degrees
    r_millimeters = r * 1e3  # Convert radius to mm
    dP_dρ = P_0 * 4 * r / w ** 2 * np.exp(-2 * (r / w) ** 2)
    dρ_dr = d * (d + R - np.sqrt(R ** 2 - r ** 2) - (2 * r**2) / (np.sqrt(R**2 - r**2))) / (d + R - np.sqrt(R ** 2 - r**2)) ** 2
    dP_dr = dP_dρ * dρ_dr
    dP_dr_normalized = dP_dr / np.trapezoid(dP_dr, r_millimeters)  # Normalize the power over the radius range

#     ax2 = plt.gca().twinx()
#     ax2.plot(β_degrees, dP_dr, linestyle='--', label='Power vs Angle of Incidence - v2')
#     ax2.set_xlabel('Angle of Incidence (degrees)')
#     ax2.set_ylabel('Power (W)')
#     ax2.grid(visible=True)
# plt.show()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Power vs Radius for a Lens\nR={R * 1e3:.2f} mm', fontsize=16)
    ax1.plot(r_millimeters, dP_dr_normalized, label='Power vs Radius', color='tab:blue')
    ax1.set_xlabel('Radius (mm)')
    ax1.set_ylabel('Power (normalized)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(r_millimeters, β_degrees, label='Angle of Incidence vs Radius', linestyle='--', color='tab:orange')
    ax2.plot(r_millimeters, θ_degrees, label='Surface Inclination vs Radius', linestyle=':', color='tab:green')
    ax2.set_ylabel('Angle (degrees)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    ax1.grid()
    # plt.savefig(f'outputs/figures/power_vs_radius_{R * 1e5:.0f}.svg', bbox_inches='tight')
    plt.show()
# %% Total reflected power:
measured_reflectance = pd.read_csv('data/Sapphire FiveNine reflectance measurement 260323.csv')
measured_reflectance['reflection_calibrated_uW'] = measured_reflectance['reflection_uW'] - measured_reflectance['dark_uW']
measured_reflectance['R'] = measured_reflectance['reflection_calibrated_uW'] / measured_reflectance['input_power_uW']
measured_reflectance.sort_values(['polarization', 'angle_deg'], inplace=True)
s_mask = measured_reflectance['polarization'] == 's'
p_mask = measured_reflectance['polarization'] == 'p'
df_s = measured_reflectance[s_mask]
df_p = measured_reflectance[p_mask]

plt.scatter(np.abs(df_s['angle_deg']), df_s['R'])
plt.scatter(np.abs(df_p['angle_deg']), df_p['R'])
plt.xlabel('Angle of Incidence (degrees)')
plt.ylabel('Reflectance')
plt.yscale('log')
plt.title('Measured Reflectance vs Angle of Incidence for Sapphire')
plt.legend(['s-polarization', 'p-polarization'])
plt.grid()
plt.savefig(f'outputs/figures/reflected_power_vs_angle.svg', bbox_inches='tight')
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
plt.title('Measured Reflectance vs Angle of Incidence for Sapphire\n(recentered to be symmetric around 0 degrees)')
plt.legend(['s-polarization', 'p-polarization'])
plt.grid()
plt.savefig(f'outputs/figures/reflected_power_vs_angle_recentered_abs.svg', bbox_inches='tight')
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

fig, ax = plt.subplots(3, 1, figsize=(10, 18))

ax[0].scatter(np.abs(df_s['angle_deg']), df_s['R'])
ax[0].scatter(np.abs(df_p['angle_deg']), df_p['R'])
ax[0].set_xlabel('Absolute Angle of Incidence (degrees)')
ax[0].set_ylabel('Reflectance')
ax[0].set_yscale('log')
ax[0].set_title('Measured Reflectance vs Absolute Angle of Incidence for Sapphire')
ax[0].legend(['s-polarization', 'p-polarization'])
ax[0].grid()

ax[1].scatter(np.abs(df_s_calibrated['angle_deg']), df_s_calibrated['R'])
ax[1].scatter(np.abs(df_p_calibrated['angle_deg']), df_p_calibrated['R'])
ax[1].set_xlabel('Angle of Incidence (degrees)')
ax[1].set_ylabel('Reflectance')
ax[1].set_yscale('log')
ax[1].set_title('Measured Reflectance vs Angle of Incidence for Sapphire\n(recentered to be symmetric around 0 degrees)')
ax[1].legend(['s-polarization', 'p-polarization'])
ax[1].grid()

ax[2].scatter(np.abs(df_s_calibrated['angle_deg']), reflectance_s)
ax[2].scatter(np.abs(df_p_calibrated['angle_deg']), reflectance_p)
ax[2].set_xlabel('Angle of Incidence (degrees)')
ax[2].set_ylabel('Reflectance (Smoothed)')
ax[2].set_yscale('log')
ax[2].set_title('Measured Reflectance vs Angle of Incidence for Sapphire (recentered and Smoothed)')
ax[2].legend(['s-polarization', 'p-polarization'])
ax[2].grid()
plt.savefig(f'outputs/figures/reflected_power_vs_angle_recentered_smoothed_abs.svg', bbox_inches='tight')
plt.show()

# %%
R_s = scipy.interpolate.interp1d(np.abs(df_s_calibrated['angle_deg']), reflectance_s, kind='linear', fill_value='extrapolate')
R_p = scipy.interpolate.interp1d(np.abs(df_p_calibrated['angle_deg']), reflectance_p, kind='linear', fill_value='extrapolate')

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

total_reflected_power_s = 0.5 * np.trapezoid(dP_dβ_normalized_interpolated(beta_degrees) * R_s(beta_degrees), beta_degrees)
total_reflected_power_p = 0.5 * np.trapezoid(dP_dβ_normalized_interpolated(beta_degrees) * R_p(beta_degrees), beta_degrees)
total_reflected_power = total_reflected_power_s + total_reflected_power_p
print(f'Total Reflected Power for s-polarization: {total_reflected_power_s:.3e}')
print(f'Total Reflected Power for p-polarization: {total_reflected_power_p:.3e}')


fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(2, 2)

ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])
ax_bottom = fig.add_subplot(gs[1, :])  # span both columns

angles = np.linspace(0, 30, 1000)
ax01.plot(angles, R_s(angles))
ax01.plot(angles, R_p(angles))
ax01.set_xlabel('Angle of Incidence (degrees)')
ax01.set_ylabel('Reflectance')
ax01.set_yscale('log')
ax01.set_title('Measured Reflectance vs Angle of Incidence for Sapphire\n(Recentered, Smoothed and Interpolated)')
ax01.legend(['s-polarization', 'p-polarization'])
ax01.grid()

ax00.plot(beta_degrees, dP_dβ_normalized, label='Total Power vs Angle of Incidence')
ax00.set_xlabel('Angle of Incidence (degrees)')
ax00.set_ylabel('Power (Normalized)')
ax00.set_title(f'Total Power vs Angle of Incidence for a Lens\nR={R * 1e3:.2f} mm')
ax00.grid()
ax00.legend()

ax_bottom.plot(beta_degrees, dP_dβ_normalized_interpolated(beta_degrees) * R_s(beta_degrees))
ax_bottom.plot(beta_degrees, dP_dβ_normalized_interpolated(beta_degrees) * R_p(beta_degrees))
ax_bottom.set_xlabel('Angle of Incidence (degrees)')
ax_bottom.set_ylabel('Power (Normalized and Weighted by Reflectance)')
ax_bottom.set_title('Total Reflected Power vs Angle of Incidence')
ax_bottom.grid()
ax_bottom.legend(['s-polarization', 'p-polarization'])


plt.suptitle(f"\n\ntotal reflected power = $\\frac{{1}}{{2}}\int_{{0}}^{{\\beta_{{\max}}}}P\left(\\beta\\right)\cdot R_{{p}}\left(\\beta\\right)+P\left(\\beta\\right)\cdot R_{{s}}\left(\\beta\\right)d\\beta = {total_reflected_power:.3e}$", y=1.02)
plt.tight_layout()
plt.savefig(f'outputs/figures/total_reflected_power_analysis.svg', bbox_inches='tight')
plt.show()




