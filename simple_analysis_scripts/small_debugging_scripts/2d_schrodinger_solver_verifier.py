from simple_analysis_scripts.potential_analysis.analyze_potential import *

import numpy as np
import matplotlib.pyplot as plt


# Parameters
m = 3.0
omega = 2.0
r_max = 3 * np.sqrt(H_BAR / m * omega)  # Choose r_max based on the classical turning point
n = 1000

# Harmonic potential
def V_ho(r):
    return 0.5 * m * omega**2 * r**2

# Numerical solution
r, E0_num, psi0_num, H = ground_state_2d_radial_polar(r_max, V_ho, m, n)

# Exact solution
E0_exact = omega
psi0_exact = np.sqrt(2 * m * omega / H_BAR) * np.exp(-0.5 * m * omega * r**2 / H_BAR)

# Align overall sign
if np.dot(psi0_num, psi0_exact) < 0:
    psi0_num = -psi0_num

# Energy comparison
print(f"Numerical E0 = {E0_num:.10f}")
print(f"Exact     E0 = {E0_exact:.10f}")
print(f"Absolute error = {abs(E0_num - E0_exact):.3e}")

# Normalization check
norm = np.trapezoid(np.abs(psi0_num)**2 * r, r)
print(f"Radial normalization = {norm:.10f}")

# Wavefunction error
wavefunc_error = np.sqrt(np.trapezoid((psi0_num - psi0_exact)**2 * r, r))
print(f"Wavefunction error = {wavefunc_error:.3e}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(r, psi0_num, label=r"Numerical $\psi_0(r)$")
plt.plot(r, psi0_exact, "--", label=r"Exact $\psi_0(r)$")
plt.xlabel("r")
plt.ylabel(r"$\psi_0(r)$")
plt.title("2D Harmonic Oscillator Ground State")
plt.legend()
plt.grid(True)
plt.show()