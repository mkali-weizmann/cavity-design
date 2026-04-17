from simple_analysis_scripts.potential_analysis.analyze_potential import *
# --- use the function from before ---
# ground_state_2d_radial_polar(r_max, V, m, n)


# Parameters
m = 1.0
omega = 1.0
r_max = 8.0
n = 2000

# Harmonic potential
def V_ho(r):
    return 0.5 * m * omega**2 * r**2

# Numerical solution
r, E0_num, psi0_num, H = ground_state_2d_radial_polar(r_max, V_ho, m, n)

# Exact solution
E0_exact = omega
psi0_exact = np.sqrt(2 * m * omega) * np.exp(-0.5 * m * omega * r**2)

# Compare energies
print(f"Numerical E0 = {E0_num:.10f}")
print(f"Exact     E0 = {E0_exact:.10f}")
print(f"Absolute error = {abs(E0_num - E0_exact):.3e}")

# Compare wavefunctions
# Since eigenvectors are defined up to an overall sign, align sign if needed
if np.dot(psi0_num, psi0_exact) < 0:
    psi0_num = -psi0_num

# L2-like grid error for the radial normalization measure
wavefunc_error = np.sqrt(np.trapezoid((psi0_num - psi0_exact)**2 * r, r))
print(f"Wavefunction error = {wavefunc_error:.3e}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(r, psi0_num, label="Numerical $\\psi_0(r)$")
plt.plot(r, psi0_exact, "--", label="Exact $\\psi_0(r)$")
plt.xlabel("r")
plt.ylabel("$\\psi_0(r)$")
plt.title("2D Harmonic Oscillator Ground State")
plt.legend()
plt.grid(True)
plt.show()