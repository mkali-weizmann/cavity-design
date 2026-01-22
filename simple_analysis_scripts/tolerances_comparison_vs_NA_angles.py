from cavity import *
NA = np.linspace(0.02, 0.16, 100)
L = 0.01
unconcentricity = 4 * LAMBDA_0_LASER ** 2 / (L * np.pi**2 * NA**4)
unconcentricity_mirror_lens_mirror = unconcentricity * 41  # arms' lengths ratio + 1
R = 5e-3
y_tolerance_fabry_perot = unconcentricity * np.tan(NA * 0.46)
alpha_tolerance_fabry_perot = unconcentricity * np.tan(NA *0.46) / R
y_tolerance_mirror_lens_mirror = unconcentricity_mirror_lens_mirror * np.tan(NA * 0.46)
alpha_tolerance_mirror_lens_mirror = unconcentricity_mirror_lens_mirror * np.tan(NA * 0.46) / R

import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.plot(NA, alpha_tolerance_fabry_perot, label='Fabry-Perot Mirror', color='blue')
plt.plot(NA, alpha_tolerance_mirror_lens_mirror, label='Mirror-Lens-Mirror Small Mirror/Lens', color='orange')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Numerical Aperture')
plt.ylabel('Tilt Tolerance [rad]')
plt.title('Tilt Angle Tolerance vs Numerical Aperture')
plt.legend()
plt.grid()
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
plt.savefig(rf'outputs/figures/tolerance_vs_NA_comparison_angles-lateral_shift{timestamp}.svg')
plt.show()

plt.figure(figsize=(5, 5))
plt.plot(NA, y_tolerance_fabry_perot, label='Fabry-Perot Mirror', color='blue')
plt.plot(NA, y_tolerance_mirror_lens_mirror, label='Mirror-Lens-Mirror Small Mirror', color='orange')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Numerical Aperture (NA)')
plt.ylabel('Lateral Shift Tolerance [m]')
plt.title('Lateral Shift Tolerance vs Numerical Aperture')
plt.legend()
plt.grid()
plt.savefig(rf'outputs/figures/tolerance_vs_NA_comparison_angles-rotation{timestamp}.svg')
plt.show()