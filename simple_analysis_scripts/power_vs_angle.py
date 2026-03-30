import numpy as np
import pandas as pd
# from matplotlib import use
# use('TkAgg')  # Use a non-interactive backend for saving figures
import scipy


import matplotlib.pyplot as plt

# NOTE:
# All expressions here are the results of my derivation at the https://mynotebook.labarchives.com/MTM3NjE3My41fDEwNTg1OTUvMTA1ODU5NS9Ob3RlYm9vay8zMjQzMzA0MzY1fDM0OTMzNjMuNQ==/page/11290221-151
# Or at my research the lyx file under subsection "Power vs angle and radius"
# %%

λ = 1064e-9
P_0 = 2.8e4
Rs = (24.22e-3, 5.49e-3)
NAs = (0.156, 6.6e-3)
ds = (5e-3, 15e-2)
D = 7.75e-3  # Diameter of the lens
n_1 = 1.45
n_2 = 1.8
β_value = np.linspace(0, 32 / 360 * 2 * np.pi, 100)  # 32 degrees is the edge of the reflectance measurement,
# but some distributions of energy reach those values of incidence angles. The returned value for those reflectances is
# just interpolation of the measure values, which is not too bad because anyways those are small power.

def α_spherical(β_value, R, d):
    return np.arcsin(R / (R + d) * np.sin(β_value))

def dα_dβ_spherical(β_value, R, d):
    return R / (R + d) * np.cos(β_value) / np.sqrt(1 - (R / (R + d) * np.sin(β_value)) ** 2)

def ρ(α_value, d):
    return d * np.tan(α_value)

def dP_dρ(ρ_value, w_value):
    return P_0 * 4 * ρ_value / w_value ** 2 * np.exp(-2 * (ρ_value / w_value) ** 2)

def α_aspheric(β_value, n):
    return 2 * (β_value - np.arcsin(np.sin(β_value) / n))

def dα_dβ_aspheric(β_value, n):
    return 2 - 2 * np.cos(β_value) / np.sqrt(n ** 2 - np.sin(β_value) ** 2)

def α_D_focused(β_value):
    return β_value

def dα_dβ_D_focused(β_value):
    return 1

def α_D_collimated(β_value, n):
    q = np.sin(β_value) * (np.sqrt(np.maximum(n ** 2 - np.sin(β_value) ** 2, 0.0)) - np.cos(β_value))
    α = np.arcsin(q)
    return α

def dα_dβ_D_collimated(β_value, n):
    s = np.sin(β_value)
    c = np.cos(β_value)

    # Avoid negative due to numerical noise
    root_expression = np.sqrt(np.maximum(n ** 2 - s ** 2, 0.0))

    # NA(β)
    NA = s * (root_expression - c)

    # dNA/dβ
    dNA_dbeta = (
            s * (s - (c * s) / root_expression)
            + c * (root_expression - c)
    )

    # dα/dβ
    denom = np.sqrt(np.maximum(1.0 - NA ** 2, 0.0))

    return dNA_dbeta / denom

def generate_power_vs_aoi_distribution(β_value, NA, d, α_function, dα_dβ_function, **α_kwargs):
    w_0 = λ / (np.pi * NA)
    z_R = np.pi * w_0 ** 2 / λ
    w = w_0 * np.sqrt(1 + (d / z_R) ** 2)
    if α_function == α_spherical:
        α_value = α_function(β_value, R=α_kwargs['R'], d=d)
    elif α_function in (α_aspheric, α_D_collimated):
        α_value = α_function(β_value, n=α_kwargs['n'])
    else:
        α_value = α_function(β_value)
    ρ_value = ρ(α_value, d)
    dP_dρ_value = dP_dρ(ρ_value, w)
    dρ_dα = d / np.cos(α_value) ** 2
    if dα_dβ_function == dα_dβ_spherical:
        dα_dβ = dα_dβ_function(β_value, R=α_kwargs['R'], d=d)
    elif dα_dβ_function in (dα_dβ_aspheric, dα_dβ_D_collimated):
        dα_dβ = dα_dβ_function(β_value, n=α_kwargs['n'])
    else:
        dα_dβ = dα_dβ_function(β_value)
    dP_dβ = dP_dρ_value * dρ_dα * dα_dβ
    beta_degrees = β_value / (2 * np.pi) * 360  # Convert to degrees
    dP_dβ_normalized = dP_dβ / np.trapezoid(np.where(np.isnan(dP_dβ), 0, dP_dβ), beta_degrees)  # Normalize the power over the angle range
    return beta_degrees, dP_dβ_normalized

beta_degrees, dP_dβ_high_NA_side = generate_power_vs_aoi_distribution(β_value, NA=NAs[0], d=ds[0], α_function=α_spherical, dα_dβ_function=dα_dβ_spherical, R=Rs[0])
_, dP_dβ_low_NA_side = generate_power_vs_aoi_distribution(β_value, NA=NAs[1], d=ds[1], α_function=α_spherical, dα_dβ_function=dα_dβ_spherical, R=Rs[1])
_, dP_dβ_best_form_n_1 = generate_power_vs_aoi_distribution(β_value, NA=NAs[0], d=ds[0], α_function=α_aspheric, dα_dβ_function=dα_dβ_aspheric, n=n_1)
_, dP_dβ_best_form_n_2 = generate_power_vs_aoi_distribution(β_value, NA=NAs[0], d=ds[0], α_function=α_aspheric, dα_dβ_function=dα_dβ_aspheric, n=n_2)
_, dP_dβ_best_form_D_collimated_n_1 = generate_power_vs_aoi_distribution(β_value, NA=NAs[0], d=ds[0], α_function=α_D_collimated, dα_dβ_function=dα_dβ_D_collimated, n=n_1)
_, dP_dβ_best_form_D_collimated_n_2 = generate_power_vs_aoi_distribution(β_value, NA=NAs[0], d=ds[0], α_function=α_D_collimated, dα_dβ_function=dα_dβ_D_collimated, n=n_2)
_, dP_dβ_best_form_D_focused = generate_power_vs_aoi_distribution(β_value, NA=NAs[0], d=ds[0], α_function=α_D_focused, dα_dβ_function=dα_dβ_D_focused)

plt.figure(figsize=(10, 6))
plt.plot(beta_degrees, dP_dβ_high_NA_side, label='LaserOptik - high NA side')
plt.plot(beta_degrees, dP_dβ_low_NA_side, label='LaserOptik - low NA side')
plt.plot(beta_degrees, dP_dβ_best_form_n_1, linestyle='--', label=f'Best form aspheric - n={n_1}')
plt.plot(beta_degrees, dP_dβ_best_form_n_2, linestyle=':', label=f'Best form aspheric - n={n_2}')
plt.plot(beta_degrees, dP_dβ_best_form_D_collimated_n_1, linestyle='-.', label=f'D aspheric, ) side - n={n_1}')
plt.plot(beta_degrees, dP_dβ_best_form_D_collimated_n_2, linestyle=(0, (5, 1)), label=f'D aspheric, ) side - n={n_2}')
plt.plot(beta_degrees, dP_dβ_best_form_D_focused, linestyle=(0, (3, 1, 1, 1)), label='D aspheric, | side')
plt.xlabel('Angle of Incidence (degrees)')
plt.ylabel('Power (Normalized)')
plt.xlim(0, 40)
plt.title(f'Total Power vs Angle of Incidence for a Lens\nR={Rs[0] * 1e3:.2f} mm')
plt.grid()
plt.legend()
# plt.savefig(f'figures/total_power_vs_angle_{R * 1e5:.0f}.svg', bbox_inches='tight')
plt.show()

# save dP_dβ_normalized and beta_angles to a csv file:
# import pandas as pd
# df = pd.DataFrame({'Angle of Incidence (degrees)': beta_angles, 'Total Power (W)': dP_dβ_normalized})
# df.to_csv(f'figures/total_power_vs_angle_{R * 1e5:.0f}.csv', index=False)
# print(f'Saved data for R={R * 1e3:.2f} mm to figures/total_power_vs_angle_{R * 1e5:.0f}.csv')

# %% Total reflected power:
measured_reflectance = pd.read_csv('data/Sapphire FiveNine reflectance measurement 260323.csv')
measured_reflectance['reflection_calibrated_uW'] = measured_reflectance['reflection_uW'] - measured_reflectance['dark_uW']
measured_reflectance['R'] = measured_reflectance['reflection_calibrated_uW'] / measured_reflectance['input_power_uW']
measured_reflectance.sort_values(['polarization', 'angle_deg'], inplace=True)
s_mask = measured_reflectance['polarization'] == 's'
p_mask = measured_reflectance['polarization'] == 'p'
df_s = measured_reflectance[s_mask]
df_p = measured_reflectance[p_mask]

# IMPORTANT: DIVIDE REFLECTANCE BY 2 BECAUSE THE REFLECTANCE HAPPENED TWICE - BOTH AT THE FIRST AND SECOND FACE OF THE SAMPLE
df_s.loc[:,'R'] = df_s['R'] / 2
df_p.loc[:,'R'] = df_p['R'] / 2

plt.close('all')
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

# %% Recenter data (it is somewhy shifted)
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

# %% Smooth data with a Gaussian kernel to reduce noise and make the reflectance curves more visually interpretable
def gaussian_smooth(x, y, sigma):
    y_out = np.empty_like(y)

    for i, xi in enumerate(x):
        w = np.exp(-(x - xi)**2 / (2*sigma**2))
        y_out[i] = np.sum(w*y) / np.sum(w)

    return y_out

reflectance_s = gaussian_smooth(np.abs(df_s_calibrated['angle_deg'].to_numpy()), df_s_calibrated['R'].to_numpy(), sigma=1)
reflectance_p = gaussian_smooth(np.abs(df_p_calibrated['angle_deg'].to_numpy()), df_p_calibrated['R'].to_numpy(), sigma=1)

plt.close('all')
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

# %% Interpolate the smoothed reflectance data to get a continuous function of reflectance vs angle of incidence, which can be used for integration later
R_s = scipy.interpolate.interp1d(np.abs(df_s_calibrated['angle_deg']), reflectance_s, kind='linear', fill_value='extrapolate')
R_p = scipy.interpolate.interp1d(np.abs(df_p_calibrated['angle_deg']), reflectance_p, kind='linear', fill_value='extrapolate')

def integrated_reflectances(dP_dβ_normalized, beta_degrees):
    total_reflected_power_s = 0.5 * np.trapezoid(dP_dβ_normalized * R_s(beta_degrees), beta_degrees)
    total_reflected_power_p = 0.5 * np.trapezoid(dP_dβ_normalized * R_p(beta_degrees), beta_degrees)
    total_reflected_power = total_reflected_power_s + total_reflected_power_p
    return total_reflected_power, total_reflected_power_s, total_reflected_power_p


# %% Conclusion cell
total_reflected_power_avantier, total_reflected_power_s_avantier, total_reflected_power_p_avantier = integrated_reflectances(dP_dβ_high_NA_side, beta_degrees)
print(f'Total Reflected Power for s-polarization: {total_reflected_power_s_avantier:.3e}')
print(f'Total Reflected Power for p-polarization: {total_reflected_power_p_avantier:.3e}')

total_reflected_power_aspheric_n_1, total_reflected_power_s_aspheric_n_1, total_reflected_power_p_aspheric_n_1 = integrated_reflectances(dP_dβ_best_form_n_1, beta_degrees)
print(f'Total Reflected Power for s-polarization: {total_reflected_power_s_aspheric_n_1:.3e}')
print(f'Total Reflected Power for p-polarization: {total_reflected_power_p_aspheric_n_1:.3e}')

total_reflected_power_aspheric_n_2, total_reflected_power_s_aspheric_n_2, total_reflected_power_p_aspheric_n_2 = integrated_reflectances(dP_dβ_best_form_n_2, beta_degrees)
print(f'Total Reflected Power for s-polarization: {total_reflected_power_s_aspheric_n_2:.3e}')
print(f'Total Reflected Power for p-polarization: {total_reflected_power_p_aspheric_n_2:.3e}')

total_reflected_power_D_collimated_n_1, total_reflected_power_s_D_collimated_n_1, total_reflected_power_p_D_collimated_n_1 = integrated_reflectances(dP_dβ_best_form_D_collimated_n_1, beta_degrees)
print(f'Total Reflected Power for s-polarization: {total_reflected_power_s_D_collimated_n_1:.3e}')
print(f'Total Reflected Power for p-polarization: {total_reflected_power_p_D_collimated_n_1:.3e}')

total_reflected_power_D_collimated_n_2, total_reflected_power_s_D_collimated_n_2, total_reflected_power_p_D_collimated_n_2 = integrated_reflectances(dP_dβ_best_form_D_collimated_n_2, beta_degrees)
print(f'Total Reflected Power for s-polarization: {total_reflected_power_s_D_collimated_n_2:.3e}')
print(f'Total Reflected Power for p-polarization: {total_reflected_power_p_D_collimated_n_2:.3e}')

total_reflected_power_D_focused, total_reflected_power_s_D_focused, total_reflected_power_p_D_focused = integrated_reflectances(dP_dβ_best_form_D_focused, beta_degrees)
print(f'Total Reflected Power for s-polarization: {total_reflected_power_s_D_focused:.3e}')
print(f'Total Reflected Power for p-polarization: {total_reflected_power_p_D_focused:.3e}')

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

ax00.plot(beta_degrees, dP_dβ_high_NA_side, color='blue', label='Avantier one side')
ax00.plot(beta_degrees, dP_dβ_best_form_n_1, color='orange', label='perfect aspheric n=1.45')
ax00.plot(beta_degrees, dP_dβ_best_form_n_2, color='green', label='perfect aspheric n=1.8')
ax00.plot(beta_degrees, dP_dβ_best_form_D_collimated_n_1, color='purple', label='perfect D-collimated n=1.45')
ax00.plot(beta_degrees, dP_dβ_best_form_D_collimated_n_2, color='brown', label='perfect D-collimated n=1.8')
ax00.plot(beta_degrees, dP_dβ_best_form_D_focused, color='gray', label='perfect D-focused')
ax00.set_xlabel('Angle of Incidence (degrees)')
ax00.set_ylabel('Power (Normalized)')
ax00.set_title(f'Total Power vs Angle of Incidence for a Lens')
ax00.grid()
ax00.legend()

ax_bottom.plot(beta_degrees, dP_dβ_high_NA_side * R_s(beta_degrees), color='blue', linestyle='-', label='s-polarization Avantier')
ax_bottom.plot(beta_degrees, dP_dβ_high_NA_side * R_p(beta_degrees), color='blue', linestyle='--', label='p-polarization Avantier')
ax_bottom.plot(beta_degrees, dP_dβ_best_form_n_1 * R_s(beta_degrees), color='orange', linestyle='-', label='s-polarization aspheric n=1.45')
ax_bottom.plot(beta_degrees, dP_dβ_best_form_n_1 * R_p(beta_degrees), color='orange', linestyle='--', label='p-polarization aspheric n=1.45')
ax_bottom.plot(beta_degrees, dP_dβ_best_form_n_2 * R_s(beta_degrees), color='green', linestyle='-', label='s-polarization aspheric n=1.8')
ax_bottom.plot(beta_degrees, dP_dβ_best_form_n_2 * R_p(beta_degrees), color='green', linestyle='--', label='p-polarization aspheric n=1.8')
ax_bottom.plot(beta_degrees, dP_dβ_best_form_D_collimated_n_1 * R_s(beta_degrees), color='purple', linestyle='-', label='s-polarization D-collimated n=1.45')
ax_bottom.plot(beta_degrees, dP_dβ_best_form_D_collimated_n_1 * R_p(beta_degrees), color='purple', linestyle='--', label='p-polarization D-collimated n=1.45')
ax_bottom.plot(beta_degrees, dP_dβ_best_form_D_collimated_n_2 * R_s(beta_degrees), color='brown', linestyle='-', label='s-polarization D-collimated n=1.8')
ax_bottom.plot(beta_degrees, dP_dβ_best_form_D_collimated_n_2 * R_p(beta_degrees), color='brown', linestyle='--', label='p-polarization D-collimated n=1.8')
ax_bottom.plot(beta_degrees, dP_dβ_best_form_D_focused * R_s(beta_degrees), color='gray', linestyle='-', label='s-polarization D-focused')
ax_bottom.plot(beta_degrees, dP_dβ_best_form_D_focused * R_p(beta_degrees), color='gray', linestyle='--', label='p-polarization D-focused')
ax_bottom.set_xlabel('Angle of Incidence (degrees)')
ax_bottom.set_ylabel('Power (Normalized and Weighted by Reflectance)')
ax_bottom.set_title('Total Reflected Power vs Angle of Incidence')
ax_bottom.grid()
ax_bottom.legend()


plt.suptitle(
    f"\n\ntotal reflected power = $\\frac{{1}}{{2}}\\int_{{0}}^{{\\beta_{{\max}}}}P\\left(\\beta\\right)\\cdot R_{{p}}\\left(\\beta\\right)+P\\left(\\beta\\right)\\cdot R_{{s}}\\left(\\beta\\right)d\\beta =$\n Avantier: {total_reflected_power_avantier:.3e}, aspheric 1.45: {total_reflected_power_aspheric_n_1:.3e}, aspheric 1.8: {total_reflected_power_aspheric_n_2:.3e}, D-collimated 1.45: {total_reflected_power_D_collimated_n_1:.3e}, D-collimated 1.8: {total_reflected_power_D_collimated_n_2:.3e}, D-focused: {total_reflected_power_D_focused:.3e}",
    y=0.99,
    fontsize=10,
)
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(f'outputs/figures/total_reflected_power_analysis.svg', bbox_inches='tight')
plt.show()
# %% Losses per NA:
NA_array = np.linspace(0.05, 0.2, 100)
losses_per_NA = np.zeros((len(NA_array), 6))
for i, NA in enumerate(NA_array):
    beta_degrees, dP_dβ_high_NA_side = generate_power_vs_aoi_distribution(β_value, NA=NA, d=ds[0], α_function=α_spherical, dα_dβ_function=dα_dβ_spherical, R=Rs[0])
    _, dP_dβ_best_form_n_1 = generate_power_vs_aoi_distribution(β_value, NA=NA, d=ds[0], α_function=α_aspheric, dα_dβ_function=dα_dβ_aspheric, n=n_1)
    _, dP_dβ_best_form_n_2 = generate_power_vs_aoi_distribution(β_value, NA=NA, d=ds[0], α_function=α_aspheric, dα_dβ_function=dα_dβ_aspheric, n=n_2)
    _, dP_dβ_best_form_D_collimated_n_1 = generate_power_vs_aoi_distribution(β_value, NA=NA, d=ds[0], α_function=α_D_collimated, dα_dβ_function=dα_dβ_D_collimated, n=n_1)
    _, dP_dβ_best_form_D_collimated_n_2 = generate_power_vs_aoi_distribution(β_value, NA=NA, d=ds[0], α_function=α_D_collimated, dα_dβ_function=dα_dβ_D_collimated, n=n_2)
    _, dP_dβ_best_form_D_focused = generate_power_vs_aoi_distribution(β_value, NA=NA, d=ds[0], α_function=α_D_focused, dα_dβ_function=dα_dβ_D_focused)

    total_reflected_power_avantier, total_reflected_power_s_avantier, total_reflected_power_p_avantier = integrated_reflectances(dP_dβ_high_NA_side, beta_degrees)
    total_reflected_power_aspheric_n_1, total_reflected_power_s_aspheric_n_1, total_reflected_power_p_aspheric_n_1 = integrated_reflectances(dP_dβ_best_form_n_1, beta_degrees)
    total_reflected_power_aspheric_n_2, total_reflected_power_s_aspheric_n_2, total_reflected_power_p_aspheric_n_2 = integrated_reflectances(dP_dβ_best_form_n_2, beta_degrees)
    total_reflected_power_D_collimated_n_1, total_reflected_power_s_D_collimated_n_1, total_reflected_power_p_D_collimated_n_1 = integrated_reflectances(dP_dβ_best_form_D_collimated_n_1, beta_degrees)
    total_reflected_power_D_collimated_n_2, total_reflected_power_s_D_collimated_n_2, total_reflected_power_p_D_collimated_n_2 = integrated_reflectances(dP_dβ_best_form_D_collimated_n_2, beta_degrees)
    total_reflected_power_D_focused, total_reflected_power_s_D_focused, total_reflected_power_p_D_focused = integrated_reflectances(dP_dβ_best_form_D_focused, beta_degrees)

    losses_per_NA[i, 0] = total_reflected_power_avantier
    losses_per_NA[i, 1] = total_reflected_power_aspheric_n_1
    losses_per_NA[i, 2] = total_reflected_power_aspheric_n_2
    losses_per_NA[i, 3] = total_reflected_power_D_collimated_n_1
    losses_per_NA[i, 4] = total_reflected_power_D_collimated_n_2
    losses_per_NA[i, 5] = total_reflected_power_D_focused
# %%
plt.figure(figsize=(10, 6))
plt.plot(NA_array, 4e6 * losses_per_NA[:, 0], label='Avantier one side (n=1.8)')
plt.plot(NA_array, 4e6 * losses_per_NA[:, 1], label='perfect aspheric n=1.45')
plt.plot(NA_array, 4e6 * losses_per_NA[:, 2], label='perfect aspheric n=1.8')
plt.plot(NA_array, 2e6 * losses_per_NA[:, 3] + 2e6 * losses_per_NA[:, 5], label='D n=1.45')
plt.plot(NA_array, 2e6 * losses_per_NA[:, 4] + 2e6 * losses_per_NA[:, 5], label='D n=1.8')
plt.ylim(0, None)
plt.xlabel('mode NA in short arm')
plt.ylabel('Total Reflected Power (ppm)')
plt.title('Roundtrip losses vs NA for different lens designs')
plt.grid()
plt.legend()
plt.savefig(f'outputs/figures/total_reflected_power_vs_NA.svg', bbox_inches='tight')
plt.show()
# %%  # Plot dalpha/dbeta * R(beta)
alpha_values_spherical = α_spherical(β_value, R=Rs[0], d=ds[0])
dalpha_dbeta_spherical = dα_dβ_spherical(β_value, R=Rs[0], d=ds[0])
alpha_values_aspheric_n_1 = α_aspheric(β_value, n=n_1)
dalpha_dbeta_aspheric_n_1 = dα_dβ_aspheric(β_value, n=n_1)
alpha_values_aspheric_n_2 = α_aspheric(β_value, n=n_2)
dalpha_dbeta_aspheric_n_2 = dα_dβ_aspheric(β_value, n=n_2)
alpha_values_D_collimated_n_1 = α_D_collimated(β_value, n=n_1)
dalpha_dbeta_D_collimated_n_1 = dα_dβ_D_collimated(β_value, n=n_1)
alpha_values_D_collimated_n_2 = α_D_collimated(β_value, n=n_2)
dalpha_dbeta_D_collimated_n_2 = dα_dβ_D_collimated(β_value, n=n_2)
alpha_values_D_focused = α_D_focused(β_value)
dalpha_dbeta_D_focused = dα_dβ_D_focused(β_value)
R_s_values = R_s(beta_degrees)
R_p_values = R_p(beta_degrees)

dalpha_dbeta_R_s_spherical = dalpha_dbeta_spherical * R_s_values
dalpha_dbeta_R_p_spherical = dalpha_dbeta_spherical * R_p_values
dalpha_dbeta_R_s_aspheric_n_1 = dalpha_dbeta_aspheric_n_1 * R_s_values
dalpha_dbeta_R_p_aspheric_n_1 = dalpha_dbeta_aspheric_n_1 * R_p_values
dalpha_dbeta_R_s_aspheric_n_2 = dalpha_dbeta_aspheric_n_2 * R_s_values
dalpha_dbeta_R_p_aspheric_n_2 = dalpha_dbeta_aspheric_n_2 * R_p_values
dalpha_dbeta_R_s_D_collimated_n_1 = dalpha_dbeta_D_collimated_n_1 * R_s_values
dalpha_dbeta_R_p_D_collimated_n_1 = dalpha_dbeta_D_collimated_n_1 * R_p_values
dalpha_dbeta_R_s_D_collimated_n_2 = dalpha_dbeta_D_collimated_n_2 * R_s_values
dalpha_dbeta_R_p_D_collimated_n_2 = dalpha_dbeta_D_collimated_n_2 * R_p_values
dalpha_dbeta_R_s_D_focused = dalpha_dbeta_D_focused * R_s_values
dalpha_dbeta_R_p_D_focused = dalpha_dbeta_D_focused * R_p_values


prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
default_colors = prop_cycle.by_key().get("color")

plt.figure(figsize=(10, 6))
plt.plot(beta_degrees, 0.5e6 * dalpha_dbeta_R_s_spherical, color=default_colors[0],  label='s-polarization, spherical')
plt.plot(beta_degrees, 0.5e6 * dalpha_dbeta_R_p_spherical, color=default_colors[0], linestyle='--', label='p-polarization, spherical')
plt.plot(beta_degrees, 0.5e6 * dalpha_dbeta_R_s_aspheric_n_1, color=default_colors[1], label='s-polarization, aspheric n=1.45')
plt.plot(beta_degrees, 0.5e6 * dalpha_dbeta_R_p_aspheric_n_1, color=default_colors[1], linestyle='--', label='p-polarization, aspheric n=1.45')
plt.plot(beta_degrees, 0.5e6 * dalpha_dbeta_R_s_aspheric_n_2, color=default_colors[2], label='s-polarization, aspheric n=1.8')
plt.plot(beta_degrees, 0.5e6 * dalpha_dbeta_R_p_aspheric_n_2, color=default_colors[2], linestyle='--', label='p-polarization, aspheric n=1.8')
plt.plot(beta_degrees, 0.5e6 * dalpha_dbeta_R_s_D_collimated_n_1, color=default_colors[3], label='s-polarization, D-collimated n=1.45')
plt.plot(beta_degrees, 0.5e6 * dalpha_dbeta_R_p_D_collimated_n_1, color=default_colors[3], linestyle='--', label='p-polarization, D-collimated n=1.45')
plt.plot(beta_degrees, 0.5e6 * dalpha_dbeta_R_s_D_collimated_n_2, color=default_colors[4], label='s-polarization, D-collimated n=1.8')
plt.plot(beta_degrees, 0.5e6 * dalpha_dbeta_R_p_D_collimated_n_2, color=default_colors[4], linestyle='--', label='p-polarization, D-collimated n=1.8')
plt.plot(beta_degrees, 0.5e6 * dalpha_dbeta_R_s_D_focused, color=default_colors[5], label='s-polarization, D-focused')
plt.plot(beta_degrees, 0.5e6 * dalpha_dbeta_R_p_D_focused, color=default_colors[5], linestyle='--', label='p-polarization, D-focused')
plt.xlabel('Angle of Incidence (degrees)')
plt.ylabel('0.5 * dα/dβ * R(β) [ppm]')
plt.title('0.5 * dα/dβ * R(β) vs Angle of Incidence for Different Lens Designs and Polarizations')
plt.grid()
plt.legend()
plt.savefig(f'outputs/figures/dalpha_dbeta_times_R_vs_angle.svg', bbox_inches='tight')
plt.show()
# %% Plot R(beta(alpha))
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(alpha_values_spherical * 360 / (2 * np.pi), R_s(beta_degrees) * 1e6, color=default_colors[0], linestyle='--', label='s-polarization, spherical')
plt.plot(alpha_values_spherical * 360 / (2 * np.pi), R_p(beta_degrees) * 1e6, color=default_colors[0], label='p-polarization, spherical')
plt.plot(alpha_values_aspheric_n_1 * 360 / (2 * np.pi), R_s(beta_degrees) * 1e6, color=default_colors[1], label='s-polarization, aspheric n=1.45')
plt.plot(alpha_values_aspheric_n_1 * 360 / (2 * np.pi), R_p(beta_degrees) * 1e6, color=default_colors[1], linestyle='--', label='p-polarization, aspheric n=1.45')
plt.plot(alpha_values_aspheric_n_2 * 360 / (2 * np.pi), R_s(beta_degrees) * 1e6, color=default_colors[2], label='s-polarization, aspheric n=1.8')
plt.plot(alpha_values_aspheric_n_2 * 360 / (2 * np.pi), R_p(beta_degrees) * 1e6, color=default_colors[2], linestyle='--', label='p-polarization, aspheric n=1.8')
plt.plot(alpha_values_D_collimated_n_1 * 360 / (2 * np.pi), R_s(beta_degrees) * 1e6, color=default_colors[3], label='s-polarization, D-collimated n=1.45')
plt.plot(alpha_values_D_collimated_n_1 * 360 / (2 * np.pi), R_p(beta_degrees) * 1e6, color=default_colors[3], linestyle='--', label='p-polarization, D-collimated n=1.45')
plt.plot(alpha_values_D_collimated_n_2 * 360 / (2 * np.pi), R_s(beta_degrees) * 1e6, color=default_colors[4], label='s-polarization, D-collimated n=1.8')
plt.plot(alpha_values_D_collimated_n_2 * 360 / (2 * np.pi), R_p(beta_degrees) * 1e6, color=default_colors[4], linestyle='--', label='p-polarization, D-collimated n=1.8')
plt.plot(alpha_values_D_focused * 360 / (2 * np.pi), R_s(beta_degrees) * 1e6, color=default_colors[5], label='s-polarization, D-focused')
plt.plot(alpha_values_D_focused * 360 / (2 * np.pi), R_p(beta_degrees) * 1e6, color=default_colors[5], linestyle='--', label='p-polarization, D-focused')
plt.xlabel('Ray Inclination Angle α (degrees)')
plt.ylabel('Reflectance R(β(α)) [ppm]')
plt.title('Reflectance R vs Ray Inclination Angle α')
plt.grid()
plt.legend()
plt.savefig(f'outputs/figures/reflectance_vs_alpha.svg', bbox_inches='tight')
plt.show()

# %% Power vs radius (spherical only, not really being used)
# for i in range(0, 2):
#     R = Rs[i]
#     NA = NAs[i]
#     d = ds[i]
#     w_0 = λ / (np.pi * NA)
#     z_R = np.pi * w_0 ** 2 / λ
#     w = w_0 * np.sqrt(1 + (d / z_R) ** 2)
#     r = np.linspace(0, D / 2, 10000)
#     θ = np.arcsin(r / R)
#     α = np.arctan(r / (d + R - np.sqrt(R ** 2 - r ** 2)))
#     β_value = θ + α
#     β_degrees = β_value / (2 * np.pi) * 360  # Convert to degrees
#     θ_degrees = θ / (2 * np.pi) * 360  # Convert to degrees
#     r_millimeters = r * 1e3  # Convert radius to mm
#     dP_dρ = P_0 * 4 * r / w ** 2 * np.exp(-2 * (r / w) ** 2)
#     dρ_dr = d * (d + R - np.sqrt(R ** 2 - r ** 2) - (2 * r**2) / (np.sqrt(R**2 - r**2))) / (d + R - np.sqrt(R ** 2 - r**2)) ** 2
#     dP_dr = dP_dρ * dρ_dr
#     dP_dr_normalized = dP_dr / np.trapezoid(dP_dr, r_millimeters)  # Normalize the power over the radius range
#
# #     ax2 = plt.gca().twinx()
# #     ax2.plot(β_degrees, dP_dr, linestyle='--', label='Power vs Angle of Incidence - v2')
# #     ax2.set_xlabel('Angle of Incidence (degrees)')
# #     ax2.set_ylabel('Power (W)')
# #     ax2.grid(visible=True)
# # plt.show()
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#     fig.suptitle(f'Power vs Radius for a Lens\nR={R * 1e3:.2f} mm', fontsize=16)
#     ax1.plot(r_millimeters, dP_dr_normalized, label='Power vs Radius', color='tab:blue')
#     ax1.set_xlabel('Radius (mm)')
#     ax1.set_ylabel('Power (normalized)', color='tab:blue')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')
#
#     ax2 = ax1.twinx()
#     ax2.plot(r_millimeters, β_degrees, label='Angle of Incidence vs Radius', linestyle='--', color='tab:orange')
#     ax2.plot(r_millimeters, θ_degrees, label='Surface Inclination vs Radius', linestyle=':', color='tab:green')
#     ax2.set_ylabel('Angle (degrees)', color='tab:orange')
#     ax2.tick_params(axis='y', labelcolor='tab:orange')
#
#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
#
#     ax1.grid()
#     # plt.savefig(f'outputs/figures/power_vs_radius_{R * 1e5:.0f}.svg', bbox_inches='tight')
#     plt.show()
