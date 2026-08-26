# %%
from matplotlib import use
use("QT5Agg")  # for interactive plotting in Spyder
from cavity_design import *
cavity = OpticalSystem(
    elements=[
        SphericalMirror(name='LaserOptik mirror', radius=0.005, outwards_normal=np.array([-1.0, 0.0, 0.0]), center=np.array([-0.005, 0.0, 0.0]), curvature_sign=-1, diameter=0.00775, material_properties=MaterialProperties(refractive_index=1.45, alpha_expansion=5.2e-07, beta_surface_absorption=1e-06, kappa_conductivity=1.38, dn_dT=1.2e-05, nu_poisson_ratio=0.16, alpha_volume_absorption=0.001, intensity_reflectivity=0.0001, intensity_transmittance=0.999899, temperature=np.nan)),
        OpticalSystem(
            elements=[
                AsphericRefractiveSurface(name='Cartesian oval lens - back side (vendor asphere)', center=np.array([0.004999999999999998, 0.0, 0.0]), outwards_normal=np.array([-1.0, -0.0, -0.0]), polynomial_coefficients=np.array([0.0, 19.718915818155743, -1366961.8233454428, 66220393645.204445, -4629070442344418.0, 3.933540206193227e+20, -3.1004050267916582e+25, 1.9019455280882967e+30, -8.0953027932369e+34, 2.0766853858103462e+39, -2.3902396558132647e+43]), curvature_sign=1, n_1=1.0, n_2=1.76, diameter=0.007749999999999999, material_properties=MaterialProperties(refractive_index=1.45, alpha_expansion=5.2e-07, beta_surface_absorption=1e-06, kappa_conductivity=1.38, dn_dT=1.2e-05, nu_poisson_ratio=0.16, alpha_volume_absorption=0.001, intensity_reflectivity=0.0001, intensity_transmittance=0.999899, temperature=np.nan)),
                AsphericRefractiveSurface(name='Cartesian oval lens - front side (vendor asphere)', center=np.array([0.008699999999999998, 0.0, 0.0]), outwards_normal=np.array([1.0, 0.0, 0.0]), polynomial_coefficients=np.array([0.0, 85.6943910872071, 396788.9737967076, 3741449210.097625, 44030962754744.52, 5.818251474525176e+17, 8.328840134159115e+21, 1.0440587828746894e+26, 3.179003514185044e+30, -3.385188978050625e+34, 1.9275451029627756e+39]), curvature_sign=-1, n_1=1.76, n_2=1.0, diameter=0.007749999999999999, material_properties=MaterialProperties(refractive_index=1.45, alpha_expansion=5.2e-07, beta_surface_absorption=1e-06, kappa_conductivity=1.38, dn_dT=1.2e-05, nu_poisson_ratio=0.16, alpha_volume_absorption=0.001, intensity_reflectivity=0.0001, intensity_transmittance=0.999899, temperature=np.nan)),
            ],
            use_paraxial_ray_tracing=True,     lambda_0_laser=1.064e-06,     t_is_trivial=True,     p_is_trivial=True,     name='Cartesian oval lens (vendor asphere, corrections to rho^8)',
        ),
        SphericalMirror(radius=0.19994863712672875, outwards_normal=np.array([1.0, 0.0, 0.0]), center=np.array([0.4072830159804838, 0.0, 0.0]), curvature_sign=-1, diameter=np.nan),
    ],
         lambda_0_laser=1.064e-06,     t_is_trivial=True,     p_is_trivial=True,     use_paraxial_ray_tracing=False,
)
cavity.plot()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(3e-3, 10e-3)
ax.set_title("")
plt.show()