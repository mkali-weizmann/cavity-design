from cavity import *
from simple_analysis_scripts.small_debugging_scripts.asphere_surface_intersections_and_refraction import forwards_normal

phi = 0
theta = 0

f = 20.0
T_c = 3.0
n_1 = 1
n_2 = 1.5
polynomial_coefficients = [-5.47939897e-06, 4.54562088e-02, 4.02452659e-05, 5.53445352e-08, 6.96909906e-11]  # generated for f=20, Tc=3 in aspheric_lens_generator.py
polynomial = Polynomial(polynomial_coefficients)

optical_axis = np.array([np.cos(phi), np.sin(phi), 0])

diameter = 25
back_center = f * optical_axis
front_center = back_center + T_c * optical_axis
s_1 = FlatRefractiveSurface(outwards_normal=optical_axis, center=back_center, n_1=n_1, n_2=n_2, diameter=diameter)

s_2 = AsphericRefractiveSurface(center=front_center,
                                outwards_normal=optical_axis,
                                diameter=diameter,
                                polynomial_coefficients=polynomial,
                                n_1=n_2,
                                n_2=n_1)
s_1_automatic, s_2_automatic = Surface.from_params(generate_aspheric_lens_params(f=f,
                                                                              T_c=T_c,
                                                                              n=n_2,
                                                                              forward_normal=optical_axis,
                                                                              diameter=diameter,
                                                                              polynomial_degree=8,
                                                                              flat_faces_center=back_center,
                                                                              name="aspheric_lens_automatic"))
