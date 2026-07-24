# # %%
# from cavity_design import *
#
# T_c=2.2e-3
# BFL = 2.68e-3
# lens = OpticalSystem.from_params(generate_aspheric_lens_params(back_focal_length=BFL,
#                                                                T_c=T_c, n=1.583, forward_normal=RIGHT,
#                                                                flat_faces_center=ORIGIN, diameter=6.325e-3,
#                                                                polynomial_degree=18),
#                                  )
# EFL = focal_length_of_lens(R_1=lens[0].radius, R_2=-lens[1].radius, n=lens[0].n_2, T_c=T_c)
# lens.plot()
# # set aspect ratio equal so x and y scales match
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title(f"BFL = {BFL:.2e}, EFL = {EFL:.2e}, T_c = {T_c:.2e}")
# plt.show()
# %%


# %%
from pathlib import Path
import pandas as pd

DROPBOX_DIR = Path.home() / "Weizmann Institute Dropbox" / "Michael Kali" / "Labs Dropbox"
csv_path = (
    DROPBOX_DIR
    / "Laser Phase Plate"
    / "Daily measurements and notes"
    / "2026-07-02"
    / "after removing the sperical lens"
    / "6 15"
    / "without.csv"
)
df = pd.read_csv(csv_path, skiprows=2)

# Plot first (x) and third (y) columns as Frequency vs Transmission
import matplotlib.pyplot as plt

# convert to numeric, drop rows where conversion failed and keep only frequencies between 0.7 and 0.9
x = pd.to_numeric(df.iloc[:, 0], errors='coerce')
y = pd.to_numeric(df.iloc[:, 2], errors='coerce')
# keep only rows where both x and y are numeric and frequency is in the desired range
mask = x.notna() & y.notna() & (x >= 0.7) & (x <= 0.9)
x = x[mask]
y = y[mask]

plt.plot(x, y, linestyle='-', markersize=3)
plt.xlabel('Frequency')
plt.ylabel('transmission')
plt.title('Transmission vs Frequency')
plt.tick_params(axis='both', labelbottom=False, labelleft=False)
plt.grid(True)
plt.tight_layout()
plt.savefig(Path.home() / "Desktop" / "plot.svg", format='svg')
plt.show()

