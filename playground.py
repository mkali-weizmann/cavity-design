import matplotlib.pyplot as plt

from cavity import *

params = [
          OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'                  , x=-4.999961263669513e-03  , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  )),
          OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'                     , x=6.387599281689135e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=5.488e-03               , curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  )),
          OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'                  , x=4.078081462362321e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2e-01                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  ))
         ]


unit_vector_of_angles(np.pi/2, 0)

cavity = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=LAMBDA_0_LASER,
                            set_central_line=True,
                            set_mode_parameters=True,
                            set_initial_surface=False,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4,
                            use_paraxial_ray_tracing=False,
                            debug_printing_level=1,
                            )
first_mirror = cavity.surfaces[0]

tilt_angle_p_1 = 0.002
initial_arc_lengths = tilt_angle_p_1 * first_mirror.radius
p_1 = first_mirror.parameterization(0, -initial_arc_lengths)
theta = np.linspace(-0.002, 0.002, 50)
phi = tilt_angle_p_1 + np.linspace(-0.002, 0.002, 50)
k_vector = unit_vector_of_angles(theta=theta, phi=phi + np.pi * (1 - first_mirror.inwards_normal[0])/2)  # Assume system is alligned with x axis
THETA, PHI = np.meshgrid(theta, phi)
k_vector = unit_vector_of_angles(THETA, PHI)
rays = Ray(origin=p_1, k_vector=k_vector)
rays_history = cavity.trace_ray(rays)
cavity.plot(additional_rays=rays_history[:-1])
plt.show()

optical_paths_lengths = np.stack([r.optical_path_length for r in rays_history[:-1]], axis=0).sum(axis=0)
last_phases = np.exp(1j * 2 * np.pi / LAMBDA_0_LASER * optical_paths_lengths)
last_intersection_points_parameterization = first_mirror.get_parameterization(rays_history[-1].origin)
last_intersection_points_parameterization = np.stack(last_intersection_points_parameterization, axis=-1)
p_0_parameterization = np.array([0.0005, 0.0005])
p_t_distances = p_0_parameterization - last_intersection_points_parameterization
arc_length_distance = np.sqrt(np.sum(p_t_distances**2, axis=-1))
p_0_mask = arc_length_distance < 0.0002
k_0_1 = np.sum(last_phases[p_0_mask])
plt.xlim(-0.02, 0.0005)
plt.ylim(-0.001, 0.001)

# python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# assume last_intersection_points_parameterization shape (N, M, 2)
# and last_phases shape (N, M) are available in the namespace

# flatten data
pts = last_intersection_points_parameterization.reshape(-1, 2)
x = pts[:, 0]
y = pts[:, 1]
vals = last_phases.reshape(-1)

# mask invalid entries
finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(vals.real) & np.isfinite(vals.imag)
x = x[finite_mask]
y = y[finite_mask]
phase = np.angle(vals[finite_mask])  # in \-pi..pi\ range

# build triangulation and plot filled phase map
triang = mtri.Triangulation(x, y)
fig, ax = plt.subplots(figsize=(6, 5))
pcm = ax.tripcolor(triang, phase, cmap='twilight', shading='gouraud')  # cyclic colormap
ax.set_aspect('equal', 'box')
ax.set_xlabel('parameterization p')
ax.set_ylabel('parameterization t')
cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('phase (radians)')
cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.tight_layout()
plt.show()
