import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
use('TkAgg')  # Use a non-interactive backend for saving figures


λ = 1064e-9
P_0 = 2.8e4
Rs = (24.22e-3, 5.49e-3)
NAs = (0.156, 6.6e-3)
ds = (5e-3, 15e-2)

# % First Trial
# for i in range(2):
#     R = Rs[i]
#     NA = NAs[i]
#     d = ds[i]
#     w_0 = λ / (np.pi * NA)
#     z_R = np.pi * w_0 ** 2 / λ
#     θ = np.linspace(0, np.pi / 8.85, 1000)
#     ρ = R * np.sin(θ)
#     z = d + R * (1 - np.cos(θ))
#     α = np.arctan(ρ / z)
#     angle_of_incidence = α + θ
#     w = w_0 * np.sqrt(1 + (z / z_R) ** 2)
#     d_rho_dtheta = np.cos(α + θ) / np.cos(α)
#     P = P_0 * 4 * ρ / w ** 2 * np.exp(-2 * (ρ / w) ** 2) * d_rho_dtheta
#     angle_of_incidence_degrees = angle_of_incidence / (2 * np.pi) * 360  # Convert to degrees
#     P_normalized = P / np.trapz(P, angle_of_incidence_degrees)  # Normalize the power over the angle range
#     plt.figure(figsize=(10, 6))
#     plt.plot(angle_of_incidence_degrees, P_normalized, label='Total Power vs Angle of Incidence')
#     plt.xlabel('Angle of Incidence (degrees)')
#     plt.ylabel('Total Power')
#     plt.xlim(0, 40)
#     plt.title(f'Total Power vs Angle of Incidence for a Lens (Normalized)\nR={R * 1e3:.2f} mm')
#     plt.grid()
#     plt.legend()
#     # plt.savefig(f'figures/total_power_vs_angle_{R * 1e5:.0f}.svg', bbox_inches='tight')
#     plt.show()

# %% Third trial:
for i in range(2):
    R = Rs[i]
    NA = NAs[i]
    d = ds[i]
    w_0 = λ / (np.pi * NA)
    z_R = np.pi * w_0 ** 2 / λ
    β = np.linspace(0, np.pi / 5, 1000)
    α = np.arcsin(R / (R + d) * np.sin(β))
    w = w_0 * np.sqrt(1 + (d / z_R) ** 2)
    ρ = d * np.tan(α)
    dP_dρ = P_0 * 4 * ρ / w ** 2 * np.exp(-2 * (ρ / w) ** 2)
    dρ_dα = d / np.cos(α) ** 2
    dα_dβ = R / (R + d) * np.cos(β) / np.sqrt(1 - (R / (R + d) * np.sin(β)) ** 2)
    dP_dβ = dP_dρ * dρ_dα * dα_dβ
    beta_angles = β / (2 * np.pi) * 360  # Convert to degrees
    dP_dβ_normalized = dP_dβ / np.trapz(dP_dβ, beta_angles)  # Normalize the power over the angle range
    plt.figure(figsize=(10, 6))
    plt.plot(beta_angles, dP_dβ_normalized, label='Total Power vs Angle of Incidence')
    plt.xlabel('Angle of Incidence (degrees)')
    plt.ylabel('Power (Normalized)')
    plt.xlim(0, 40)
    plt.title(f'Total Power vs Angle of Incidence for a Lens\nR={R * 1e3:.2f} mm')
    plt.grid()
    plt.legend()
    plt.savefig(f'figures/total_power_vs_angle_{R * 1e5:.0f}.svg', bbox_inches='tight')
    plt.show()

    # save dP_dβ_normalized and beta_angles to a csv file:
    import pandas as pd
    df = pd.DataFrame({'Angle of Incidence (degrees)': beta_angles, 'Total Power (W)': dP_dβ_normalized})
    df.to_csv(f'figures/total_power_vs_angle_{R * 1e5:.0f}.csv', index=False)
    print(f'Saved data for R={R * 1e3:.2f} mm to figures/total_power_vs_angle_{R * 1e5:.0f}.csv')






