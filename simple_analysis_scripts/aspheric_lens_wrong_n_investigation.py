# %%
from cavity_design import *
from matplotlib import use
use("TkAgg")
# design wavelength is smaller than actual wavelength (780 as opposed to 1064)
# => design refractive index is larger than actual refractive index (1.583 as opposed to 1.574)
n_design = 1.583
n_actual = 1.574
back_focal_length_aspheric_design = 2.68e-3
T_c_aspheric = 2.2e-3
diameter=6.325e-3
defocus=0
OPTICAL_AXIS = RIGHT
# Construct a lens for the design wavelength:
aspheric_flat, aspheric_curved = generate_aspheric_lens(
        back_focal_length=back_focal_length_aspheric_design,
        T_c=T_c_aspheric,
        n=n_design,
        forward_normal=OPTICAL_AXIS,
        flat_faces_center=ORIGIN,
        diameter=diameter,
        polynomial_degree=24,
        name="aspheric_lens_automatic",
    )
# Change the actual refractive index, after geometry was set for the design refractive index:
aspheric_flat.n_2 = n_actual
aspheric_curved.n_1 = n_actual

back_focal_length_aspheric_actual = back_focal_length_of_lens_formula(R_1=aspheric_flat.radius,
                                                                      R_2=-aspheric_curved.radius, n=n_actual,
                                                                      T_c=T_c_aspheric)

optical_system_aspheric = OpticalSystem(
        elements=[aspheric_flat, aspheric_curved],
        t_is_trivial=True,
        p_is_trivial=True,
        given_initial_central_line=True,
        use_paraxial_ray_tracing=False,
)
# Move the two lens faces rigidly: the flat face lands at the target, the curved face keeps its offset from it.
lens_displacement = (back_focal_length_aspheric_actual + defocus) * OPTICAL_AXIS - optical_system_aspheric[0].center
optical_system_aspheric.place_element(element=optical_system_aspheric[1],
                                      position=optical_system_aspheric[1].center + lens_displacement,
                                      recalculate_optic=False)
optical_system_aspheric.place_element(element=optical_system_aspheric[0],
                                      position=optical_system_aspheric[0].center + lens_displacement,
                                      recalculate_optic=True)

optical_system = OpticalSystem([LASER_OPTIK_MIRROR, optical_system_aspheric], use_paraxial_ray_tracing=False, t_is_trivial=True, p_is_trivial=True, lambda_0_laser=LAMBDA_0_LASER)
rays_0= initialize_rays(phi_max=0.3, n_rays=8)
propagated_rays = optical_system.propagate_ray(rays_0, propagate_with_first_surface_first=False)
propagated_rays.length[-1, :] = 3
ax = optical_system.plot()
propagated_rays.plot(ax=ax)
ax.set_xlim(-0.007, 3)
ax.set_ylim(-1e-3, 1.5e-3)
plt.title("When the system is collimated, peripheral rays underfocus (good sign of aberrations)")
plt.show()
