from cavity import *

R_1 = 24.22e-3
R_2 = 5.49e-3
R_2_signed = -R_2
T_c = 2.91e-3
diameter = 7.75e-3
n=1.8
back_focal_length = back_focal_length_of_lens(R_1=R_1, R_2=-R_2, n=n, T_c=T_c)
f = focal_length_of_lens(R_1=R_1, R_2=-R_2, n=n, T_c=T_c)


back_center = back_focal_length * RIGHT
surface_0, surface_1 = Surface.from_params(
    generate_aspheric_lens_params(back_focal_length=0.0042325, T_c=T_c, n=n, forward_normal=RIGHT,
                                  flat_faces_center=back_center, diameter=diameter, polynomial_degree=8,
                                  name="aspheric_lens_automatic")
)

f_aspheric = focal_length_of_lens(R_1=np.inf, R_2 = -surface_1.radius, n=n, T_c=T_c)
print(f"{f_aspheric - f:.2e} m")