from cavity import *
from matplotlib.patches import Arc

# params = [OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'                  , x=-4.999961263669513e-03, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=-5e-03               , r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )), OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'                     , x=6.456898770493272e-03, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2.422e-02            , r_2=5.488e-03            , curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03, n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=1.76e+00             , alpha_expansion=5.5e-06              , beta_surface_absorption=1e-06                , kappa_conductivity=4.606e+01            , dn_dT=1.17e-05             , nu_poisson_ratio=3e-01                , alpha_volume_absorption=1e-02                , intensity_reflectivity=1e-04                , intensity_transmittance=9.99899e-01          , temperature=np.nan               )), OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'                  , x=3.573055489216874e-01, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2e-01                , r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]
# params = [OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'                  , x=-4.999961263669513e-03, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=-5e-03               , r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )), OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'                     , x=6.387643101310118e-03, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2.422e-02            , r_2=5.488e-03            , curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03, n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=1.76e+00             , alpha_expansion=5.5e-06              , beta_surface_absorption=1e-06                , kappa_conductivity=4.606e+01            , dn_dT=1.17e-05             , nu_poisson_ratio=3e-01                , alpha_volume_absorption=1e-02                , intensity_reflectivity=1e-04                , intensity_transmittance=9.99899e-01          , temperature=np.nan               )), OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'                  , x=4.07765186689351e-01 , y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2e-01                , r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]
params = [OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'                  , x=-4.999963188089639e-03, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=-5e-03               , r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )), OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'                     , x=6.387643101310118e-03, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2.422e-02            , r_2=5.488e-03            , curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03, n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=1.76e+00             , alpha_expansion=5.5e-06              , beta_surface_absorption=1e-06                , kappa_conductivity=4.606e+01            , dn_dT=1.17e-05             , nu_poisson_ratio=3e-01                , alpha_volume_absorption=1e-02                , intensity_reflectivity=1e-04                , intensity_transmittance=9.99899e-01          , temperature=np.nan               )), OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'                  , x=4.078647854438369e-01, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2e-01                , r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]

NA_left = 1.5800000000e-01
# Generate left arm's mirror:
w_0_left = LAMBDA_0_LASER / (np.pi * NA_left)
x_left_waist = 0
mode_left_center = np.array([x_left_waist, 0, 0])

mode_left_k_vector = np.array([1, 0, 0])

mode_left = ModeParameters(
    center=np.stack([mode_left_center, mode_left_center], axis=0),
    k_vector=mode_left_k_vector,
    w_0=np.array([w_0_left, w_0_left]),
    principle_axes=np.array([[0, 0, 1], [0, 1, 0]]),
    lambda_0_laser=LAMBDA_0_LASER,
)


cavity = Cavity.from_params(
    params,
    lambda_0_laser=LAMBDA_0_LASER,
    standing_wave=True,
    p_is_trivial=True,
    t_is_trivial=True,
    set_mode_parameters=True,
    names=["Left mirror", "Lens", "Lens", "Right mirror"],
    initial_mode_parameters=mode_left,
    power=2e4,
)

plot_mirror_lens_mirror_cavity_analysis(cavity, CA=5e-3)

# Calculate arc center and radius
thetas = np.zeros(2)
for i in range(2):
    arc_center_x = cavity.arms[1].surfaces[i].origin[0]
    arc_center_y = cavity.arms[1].surfaces[i].origin[1]
    arc_radius = cavity.arms[1].surfaces[i].radius
    arc_height = 7.75e-3
    theta = np.arcsin(arc_height / arc_radius / 2) * 360 / (2*np.pi)
    add_half_circle = ((cavity.arms[1].surfaces[i].inwards_normal[0] + 1) / 2) * 180
    plt.gca().add_patch(Arc((arc_center_x, arc_center_y),
                            theta1=-theta + add_half_circle,
                            theta2=theta + add_half_circle,
                            width=arc_radius * 2,
                            height=arc_radius * 2,
                            color='black',
                            lw=1.5))
    thetas[i] = theta
# plt.title(r'Lens and $\pm\omega\left(z\right)$ rays')

corner_left_up = cavity.arms[1].surfaces[0].origin + np.array([-cavity.arms[1].surfaces[0].radius * np.cos(np.deg2rad(thetas[0])),
                                                               cavity.arms[1].surfaces[0].radius * np.sin(np.deg2rad(thetas[0])),
                                                               0])
corner_left_down = cavity.arms[1].surfaces[0].origin + np.array([-cavity.arms[1].surfaces[0].radius * np.cos(np.deg2rad(thetas[0])),
                                                               -cavity.arms[1].surfaces[0].radius * np.sin(np.deg2rad(thetas[0])),
                                                               0])
corner_right_up = cavity.arms[1].surfaces[1].origin + np.array([cavity.arms[1].surfaces[1].radius * np.cos(np.deg2rad(thetas[1])),
                                                               cavity.arms[1].surfaces[1].radius * np.sin(np.deg2rad(thetas[1])),
                                                               0])
corner_right_down = cavity.arms[1].surfaces[1].origin + np.array([cavity.arms[1].surfaces[1].radius * np.cos(np.deg2rad(thetas[1])),
                                                               -cavity.arms[1].surfaces[1].radius * np.sin(np.deg2rad(thetas[1])),
                                                               0])

plt.plot([corner_left_up[0], corner_right_up[0]], [corner_left_up[1], corner_right_up[1]], color='black', lw=1.5)
plt.plot([corner_left_down[0], corner_right_down[0]], [corner_left_down[1], corner_right_down[1]], color='black', lw=1.5)
x_0 = (cavity.arms[1].surfaces[0].center[0] + cavity.arms[1].surface_1.center[0]) / 2
y_0 = 0
d = 7e-3
plt.xlim(x_0 - d, x_0 + d)
plt.ylim(y_0 - d, y_0 + d)
xticks = plt.gca().get_xticks()
yticks = plt.gca().get_yticks()
plt.gca().set_xticklabels([f'{(tick - cavity.arms[1].surfaces[0].center[0])*1000:.0f}' for tick in xticks])
plt.gca().set_yticklabels([f'{(tick - cavity.arms[1].surfaces[0].center[0])*1000:.0f}' for tick in yticks])
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.savefig('figures/lens_actual_dimensions_1p58.svg', bbox_inches='tight')
plt.show()