# %%
from cavity_design import *

cavity = generate_one_lens_optical_system_temp(NA=0.09, short_arm_length=7.48e-3)
ax = cavity.plot()
# ax.set_title(f"NA first = {cavity.mode_parameters[0].NA[0]:.2e}, NA_last = {cavity.mode_parameters[len(cavity.arms) // 2 - 1].NA[0]:.2e}\n"
#                      f"length first = {cavity.central_line[0].length:.2e}, NA_last = {cavity.central_line[len(cavity.arms) // 2 - 1].length:.2e}")
plt.show()
# %%
cavity.arms[2].mode_parameters_on_surface_1.spot_size