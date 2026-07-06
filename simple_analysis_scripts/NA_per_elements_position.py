from cavity_design import *
from tqdm import tqdm
from matplotlib.ticker import LogLocator, ScalarFormatter

# 10cm spherical lens
params = [
          OpticalSurfaceParams(name='Laser Optik Mirror'     ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=5e-03                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None), [
          OpticalSurfaceParams(name='aspheric_lens_automatic - flat side',surface_type='flat_refractive_surface'        , x=2.656000074902883e-03   , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , radius=0                       , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1.574e+00               , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='aspheric_lens_automatic - curved side',surface_type='aspheric_surface'               , x=4.856000074902883e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.372679656101668e-03   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.574e+00               , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=np.array([ 3.88300544e-11,  2.10732198e+02,  4.36719402e+06,  1.27960456e+11, 6.83643180e+15, -2.42583052e+21,  9.26071800e+26, -2.45228128e+32, 4.25464777e+37, -4.88779028e+42,  3.54322754e+47, -1.46376839e+52, 2.62859940e+56]))
         ], [
          OpticalSurfaceParams(name='spherical_0'            ,surface_type='curved_refractive_surface'      , x=1.173200007490288e-02   , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=1.012600291464432e-01   , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.51e+00                , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='spherical_1'            ,surface_type='curved_refractive_surface'      , x=1.608200007490287e-02   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=1.012600291464432e-01   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.51e+00                , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)],
          OpticalSurfaceParams(name='End mirror'             ,surface_type='curved_mirror'                  , x=3.717769427822063e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2e-01                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=2.54e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)]

# 25cm focal length
# params = [
#           OpticalSurfaceParams(name='Laser Optik Mirror'     ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=5e-03                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None), [
#           OpticalSurfaceParams(name='aspheric_lens_automatic - flat side',surface_type='flat_refractive_surface'        , x=2.737063035388239e-03   , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , radius=0                       , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1.574e+00               , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
#           OpticalSurfaceParams(name='aspheric_lens_automatic - curved side',surface_type='aspheric_surface'               , x=4.937063035388239e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.372679656101668e-03   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.574e+00               , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=np.array([ 3.88300544e-11,  2.10732198e+02,  4.36719402e+06,  1.27960456e+11, 6.83643180e+15, -2.42583052e+21,  9.26071800e+26, -2.45228128e+32, 4.25464777e+37, -4.88779028e+42,  3.54322754e+47, -1.46376839e+52, 2.62859940e+56]))
#          ], [
#           OpticalSurfaceParams(name='spherical_0'            ,surface_type='curved_refractive_surface'      , x=1.693706303538822e-02   , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=2.542632688301439e-01   , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.51e+00                , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
#           OpticalSurfaceParams(name='spherical_1'            ,surface_type='curved_refractive_surface'      , x=2.128706303538822e-02   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.542632688301439e-01   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.51e+00                , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)],
#           OpticalSurfaceParams(name='None'                   ,surface_type='curved_mirror'                  , x=3.882870630353883e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2e-01                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=2.54e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)]

# 8cm focal length:
# params = [
#           OpticalSurfaceParams(name='Laser Optik Mirror'     ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=5e-03                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None), [
#           OpticalSurfaceParams(name='aspheric_lens_automatic - flat side',surface_type='flat_refractive_surface'        , x=2.647262396264711e-03   , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , radius=0                       , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1.574e+00               , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
#           OpticalSurfaceParams(name='aspheric_lens_automatic - curved side',surface_type='aspheric_surface'               , x=4.847262396264712e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.372679656101668e-03   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.574e+00               , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=np.array([ 3.88300544e-11,  2.10732198e+02,  4.36719402e+06,  1.27960456e+11, 6.83643180e+15, -2.42583052e+21,  9.26071800e+26, -2.45228128e+32, 4.25464777e+37, -4.88779028e+42,  3.54322754e+47, -1.46376839e+52, 2.62859940e+56]))
#          ], [
#           OpticalSurfaceParams(name='spherical_0'            ,surface_type='curved_refractive_surface'      , x=1.68472623962647e-02    , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=8.085866228222131e-02   , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.51e+00                , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
#           OpticalSurfaceParams(name='spherical_1'            ,surface_type='curved_refractive_surface'      , x=2.11972623962647e-02    , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=8.085866228222131e-02   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.51e+00                , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)],
#           OpticalSurfaceParams(name='None'                   ,surface_type='curved_mirror'                  , x=3.501972623962647e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2e-01                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=2.54e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)]

# 15cm focal length:
# params = [
#           OpticalSurfaceParams(name='Laser Optik Mirror'     ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=5e-03                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None), [
#           OpticalSurfaceParams(name='aspheric_lens_automatic - flat side',surface_type='flat_refractive_surface'        , x=2.738549950127599e-03   , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , radius=0                       , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1.574e+00               , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
#           OpticalSurfaceParams(name='aspheric_lens_automatic - curved side',surface_type='aspheric_surface'               , x=4.938549950127599e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.372679656101668e-03   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.574e+00               , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=np.array([ 3.88300544e-11,  2.10732198e+02,  4.36719402e+06,  1.27960456e+11, 6.83643180e+15, -2.42583052e+21,  9.26071800e+26, -2.45228128e+32, 4.25464777e+37, -4.88779028e+42,  3.54322754e+47, -1.46376839e+52, 2.62859940e+56]))
#          ], [
#           OpticalSurfaceParams(name='spherical_0'            ,surface_type='curved_refractive_surface'      , x=1.693854995012761e-02   , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=1.52261836004033e-01    , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.51e+00                , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
#           OpticalSurfaceParams(name='spherical_1'            ,surface_type='curved_refractive_surface'      , x=2.128854995012761e-02   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=1.52261836004033e-01    , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.51e+00                , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)],
#           OpticalSurfaceParams(name='End mirror'             ,surface_type='curved_mirror'                  , x=3.47144823810576e-01    , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2e-01                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=2.54e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)]


cavity = Cavity.from_params(params=params, standing_wave=True,
                                    lambda_0_laser=LAMBDA_0_LASER, power=28e3, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=True, set_central_line=True, set_mode_parameters=True)
cavity.plot()
plt.title(f"NA_short = {cavity.mode_parameters[0].NA[0]:.2e}, NA_long = {cavity.mode_parameters[2].NA[0]:.2e}\nlength short = {cavity.central_line[0].length*1000:.2f}mm, length long = {cavity.central_line[cavity.last_arm_index].length*1000:.2f}mm")
plt.show()

# %% 2d map:
cavity = Cavity.from_params(params=params, standing_wave=True,
                                    lambda_0_laser=LAMBDA_0_LASER, power=28e3, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=False, set_central_line=True, set_mode_parameters=True)
long_arm_lengths = np.array([25e-2, 28e-2, 29e-2, 30e-2, 31e-2, 37e-2])
short_arm_lengths = np.linspace(7.55e-3, 7.8e-3, 100)
NAs = np.zeros(shape=(len(long_arm_lengths), len(short_arm_lengths)))
mode_spacings = np.full(shape=(len(long_arm_lengths), len(short_arm_lengths)), fill_value=np.nan)
zero_derivative_points = np.full(shape=(len(long_arm_lengths), len(short_arm_lengths)), fill_value=np.nan)
focii_to_lens = np.zeros(shape=(len(long_arm_lengths)))
for i, long_arm_length in enumerate(long_arm_lengths):  #
    cavity.place_elements(elements=cavity[-1], position=long_arm_length * RIGHT, reference_center=cavity.surfaces[2], recalculate_optic=False)
    flag=False
    for j, short_arm_length in enumerate(short_arm_lengths):
        cavity.place_elements(elements=cavity[1], position = short_arm_length * RIGHT, reference_center=cavity[0])
        NAs[i, j] = cavity.arms[0].mode_parameters.NA[0]
        mode_spacings[i, j] = cavity.mode_spacing_transversal_apparent
        # if not np.isnan(cavity.arms[0].mode_parameters.NA[0]):
        #     results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=50,
        #                                                   phi_max=np.arcsin(0.3), print_tests=False)
        #     zero_derivative_points[i, j] = results_dict["zero_derivative_point"]
        if j == 0:
            focii_to_lens[i] = cavity.surfaces[-1].center[0] - cavity.surfaces[2].center[0] - cavity.surfaces[-1].radius
        if cavity.arms[0].mode_parameters.NA[0] > 0.07 and flag is False:
            focii_to_lens[i] = cavity.surfaces[-1].center[0] - cavity.surfaces[2].center[0] - cavity.surfaces[-1].radius
            flag=True


fig, ax = plt.subplots(figsize=(10, 6))
ax2 = ax.twinx()
for i in range(len(long_arm_lengths)):
    na_line, = ax.plot(short_arm_lengths * 1e3, NAs[i, :], label=f"Long arm length = {long_arm_lengths[i]*100:.2f}cm")
    # ax.plot(short_arm_lengths * 1e3, zero_derivative_points[i, :], color=na_line.get_color(), linestyle='-.', label=f"maximally allowed NA")
    ax2.plot(short_arm_lengths * 1e3, mode_spacings[i, :] / 1e6, linestyle='--', label=f"Long arm length = {long_arm_lengths[i]*100:.2f}cm, focus-to-lens = {focii_to_lens[i] * 100:.2f}cm - Mode spacing [MHz]")
ax2.set_ylim(0, 300)
ax2.set_ylabel("Mode Spacing [MHz]")
ax.set_xlabel('Short Arm Length (mm)')
ax.set_ylabel('Short Arm Numerical Aperture')
ax.axvline(7.735, color='k', linestyle='--', linewidth=1, label='Collimation point')
ax.legend()
ax.set_ylim(0, 0.2)
ax.grid()
plt.title(f"spherical focal length = {focal_length_of_lens(R_1=cavity[2][0].radius, R_2=-cavity[2][1].radius, n=cavity[2][0].n_2, T_c=cavity[2][1].center[0] - cavity[2][0].center[0])*1000:.0f} mm")
plt.savefig("outputs/figures/NA as a function of mirrors.svg")
# plt.savefig(r'figures\NA_as_a_function_of_big_mirror')
plt.show()
#
# fig, ax = plt.subplots(figsize=(10, 6))
# for i in range(len(long_arm_lengths)):
#     ax.plot(NAs[i, :], mode_spacings[i, :] / 1e6, label=f"Long arm={long_arm_lengths[i]*1e2:.2f}cm")
# ax.set_xlabel("NA")
# ax.set_ylabel("Mode spacing [MHz]")
# ax.legend()
# ax.set_yscale('log')
# ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9], numticks=15))
# ax.yaxis.set_major_formatter(ScalarFormatter())
# ax.grid()
# plt.tight_layout()
# plt.show()


# %% mode spacing and NA:
short_arm_lengths = np.array([7.35e-3, 7.45e-3, 7.55e-3, 7.65e-3])
NAs = np.linspace(0.03, 0.15, 100)
long_arm_lengths = np.full(shape=(len(short_arm_lengths), len(NAs)), fill_value=np.nan)
mode_spacings = np.full(shape=(len(short_arm_lengths), len(NAs)), fill_value=np.nan)
for i, short_arm_length in enumerate(short_arm_lengths):
    optical_system = OpticalSystem.from_params(params=params[0:-1], use_paraxial_ray_tracing=True, lambda_0_laser=LAMBDA_0_LASER, p_is_trivial=True, t_is_trivial=True,)
    optical_system.place_elements(elements=optical_system[1], position=short_arm_length * RIGHT, reference_center=optical_system[0])
    for j, NA in enumerate(NAs):
        try:
            cavity=optical_system_to_cavity_completion(optical_system=optical_system, NA=NA, end_mirror_ROC=2e-1)
            long_arm_lengths[i, j] = cavity.surfaces[-1].center[0] - cavity.surfaces[2].center[0]
            mode_spacings[i, j] = cavity.mode_spacing_transversal_apparent
        except ValueError:
            continue


plt.close('all')
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
for i in range(len(short_arm_lengths)):
    ax1.plot(long_arm_lengths[i, :]*100, NAs, label=f"Short arm={short_arm_lengths[i]*1e3:.2f}mm - NA")
    ax2.plot(long_arm_lengths[i, :]*100, mode_spacings[i, :] / 1e6, linestyle='--', label=f"Short arm={short_arm_lengths[i]*1e3:.2f}mm - df")
ax1.legend()
ax1.grid()
ax1.set_xlabel("Large mirror to aspheric distance [cm]")
ax1.set_ylabel("NA")
ax2.set_ylabel("Mode spacing [MHz]")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
for i in range(len(short_arm_lengths)):
    ax.plot(NAs, mode_spacings[i, :] / 1e6, label=f"Short arm={short_arm_lengths[i]*1e3:.2f}mm")
ax.set_xlabel("NA")
ax.set_ylabel("Mode spacing [MHz]")
ax.legend()
ax.grid()
plt.tight_layout()
plt.show()


# %% Long arm perturbation
perturbations_large_mirror = np.linspace(-4e-2, 4e-2, 100)

NAs = np.zeros_like(perturbations_large_mirror)
long_arm_lengths = np.zeros_like(perturbations_large_mirror)
for i, perturbation_value in enumerate(perturbations_large_mirror):
    perturbation_pointer = PerturbationPointer(element_index=2, parameter_name=ParamsNames.x, perturbation_value=perturbation_value)
    perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointer)
    NAs[i] = perturbed_cavity.arms[0].mode_parameters.NA[0]
    long_arm_lengths[i] = perturbed_cavity.arms[2].central_line.length

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(long_arm_lengths * 1e2, NAs, marker='o', markersize=2)
ax.set_xlabel('Long arm length (cm)')
ax.set_ylabel('Short Arm Numerical Aperture')
ax.set_title('Short Arm Numerical Aperture as a Function of Long Arm Length')
ax.grid()
plt.tight_layout()
# plt.savefig(r'figures\NA_as_a_function_of_small_mirror')
plt.show()

# %% Short arm perturbation
perturbations_aspheric_lens = np.linspace(-5e-5, 2e-4, 100)

NAs = np.zeros_like(perturbations_aspheric_lens)
short_arm_lengths = np.zeros_like(perturbations_aspheric_lens)
for i, perturbation_value in enumerate(perturbations_aspheric_lens):
    perturbation_pointer = PerturbationPointer(element_index=0, parameter_name=ParamsNames.x, perturbation_value=perturbation_value)
    perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointer)
    NAs[i] = perturbed_cavity.arms[0].mode_parameters.NA[0]
    short_arm_lengths[i] = perturbed_cavity.arms[0].central_line.length

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(short_arm_lengths * 1e2, NAs, marker='o', markersize=2)
ax.set_xlabel('Short arm length (cm)')
ax.set_ylabel('Short Arm Numerical Aperture')
ax.set_title('Short Arm Numerical Aperture as a Function of Small arm length')
ax.grid()
plt.tight_layout()
# plt.savefig(r'figures\NA_as_a_function_of_small_mirror')
plt.show()

# %%
n_lens = 40
n_mirror = 80
lens_perturbations = np.linspace(-2e-3, 1e-3, n_lens)
big_mirrors_perturbations = np.linspace(-10e-3, 200e-3, n_mirror)
NAs = np.zeros((n_lens, n_mirror))

# # grid mapping of the NA as a function of perturbations
# for i in tqdm(range(n_lens)):
#     for j in tqdm(range(n_mirror)):
#         perturbation_pointer_1 = PerturbationPointer(element_index=1, parameter_name='x', perturbation_value=lens_perturbations[i])
#         perturbation_pointer_2 = PerturbationPointer(element_index=2, parameter_name='x', perturbation_value=big_mirrors_perturbations[j])
#         cavity_perturbed = perturb_cavity(cavity=cavity, perturbation_pointer=[perturbation_pointer_1, perturbation_pointer_2])
#         NAs[i, j] = cavity_perturbed.mode_parameters[0].NA[0]
# # %%
# # Plot the 2d map of the NAs as a function of the lens perturbations and big mirrors perturbations:
# fig, ax = plt.subplots()
# im = ax.imshow(NAs, extent=(big_mirrors_perturbations[0], big_mirrors_perturbations[-1], lens_perturbations[0], lens_perturbations[-1]), origin='lower', aspect='auto')
# ax.set_xlabel('Big mirror perturbation (m)')
# ax.set_ylabel('Lens perturbation (m)')
# ax.set_title('NA as a function of lens and big mirror perturbations')
# fig.colorbar(im, ax=ax, label='NA')
# plt.show()
# %%
params_lens_unperturbed = params[1]
results = np.zeros((n_lens, n_mirror, 3))
for i in tqdm(range(n_lens), desc='lens perturbations', position=0):
    params_lens_perturbed = copy.deepcopy(params_lens_unperturbed)
    params_lens_perturbed[0].x += lens_perturbations[i]
    params_lens_perturbed[1].x += lens_perturbations[i]
    optical_system_temp = OpticalSystem.from_params(params=[params[0], params_lens_perturbed], t_is_trivial=True, p_is_trivial=True, use_paraxial_ray_tracing=True, lambda_0_laser=LAMBDA_0_LASER)
    for j, NA in enumerate(np.linspace(0.02, 0.15, 80)):
        try:
            cavity_completed = optical_system_to_cavity_completion(optical_system=optical_system_temp, NA=NA, end_mirror_ROC=2e-1)
            short_arm_length = cavity_completed.arms[0].central_line.length
            long_arm_length = cavity_completed.arms[2].central_line.length
            results[i, j, :] = [short_arm_length, long_arm_length, NA]
        except ValueError:
            results[i, j, :] = [np.nan, np.nan, NA]

# %% Plot results

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
index_ranges = [(0, 9), (9, 25), (25, n_lens)]
sc = None
for ax, (start, end) in zip(axes, index_ranges):
    sc = ax.scatter(
        results[start:end, :, 0].ravel() * 1e3,
        results[start:end, :, 1].ravel() * 1e2,
        c=results[start:end, :, 2].ravel(),
        cmap='viridis',
        s=20,
    )
    ax.set_xlabel('Short arm length (mm)')
    ax.set_title(f'lens index {start}:{end}')
axes[0].set_ylabel('Long arm length (cm)')
fig.suptitle('Long arm length vs short arm length colored by NA')
fig.colorbar(sc, ax=axes, label='NA')
plt.show()

# %% Playground
params_lens_perturbed = copy.deepcopy(params_lens_unperturbed)
lens_thickness = params_lens_perturbed[1].x - params_lens_perturbed[0].x
for short_arm_length in [7.85e-3, 8.5e-3, 9.5e-3]:
    params_laser_optik = params[0]
    params_lens_perturbed[0].x = short_arm_length + params_laser_optik.x
    params_lens_perturbed[1].x = short_arm_length + params_laser_optik.x + lens_thickness
    optical_system_temp = OpticalSystem.from_params(params=[params[0], params_lens_perturbed], t_is_trivial=True, p_is_trivial=True, use_paraxial_ray_tracing=True, lambda_0_laser=LAMBDA_0_LASER)
    for NA in [0.02, 0.05, 0.1]:
        cavity_completed = optical_system_to_cavity_completion(optical_system=optical_system_temp, NA=NA, end_mirror_ROC=2e-1)
        short_arm_length = cavity_completed.arms[0].central_line.length
        long_arm_length = cavity_completed.arms[2].central_line.length
        plot_mirror_lens_mirror_cavity_analysis(cavity=cavity_completed)
        plt.show()

        perturbable_params_names = ['x', 'y', 'phi']
        tolerance_df = cavity_completed.generate_tolerance_dataframe(perturbable_params_names=perturbable_params_names)
        tolerance_matrix = tolerance_df.to_numpy()
        ## %%
        overlaps_series = cavity_completed.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix),
                                                                  shift_numel=50,
                                                                  perturbable_params_names=perturbable_params_names, )
        ## %%
        cavity_completed.generate_overlaps_graphs(arm_index_for_NA=0, tolerance_dataframe=tolerance_df,
                                                 overlaps_series=overlaps_series,
                                                 perturbable_params_names=perturbable_params_names)
        plt.suptitle(f"NA_short = {cavity_completed.mode_parameters[0].NA[0]:.2e}, length short = {cavity_completed.central_line[0].length*1000:.2f}mm, NA_long = {cavity_completed.mode_parameters[2].NA[0]:.2e}, length long = {cavity_completed.central_line[2].length*1000:.2f}mm")
        plt.show()
# %%
params_lens_perturbed = copy.deepcopy(params_lens_unperturbed)
lens_thickness = params_lens_perturbed[1].x - params_lens_perturbed[0].x
short_arm_length = 8.1e-3
params_laser_optik = params[0]
params_lens_perturbed[0].x = short_arm_length + params_laser_optik.x
params_lens_perturbed[1].x = short_arm_length + params_laser_optik.x + lens_thickness
optical_system_temp = OpticalSystem.from_params(params=[params[0], params_lens_perturbed], t_is_trivial=True, p_is_trivial=True, use_paraxial_ray_tracing=True, lambda_0_laser=LAMBDA_0_LASER)
NA = 0.04
cavity_completed = optical_system_to_cavity_completion(optical_system=optical_system_temp, NA=NA, end_mirror_ROC=2e-1)
print(cavity_completed)
plot_mirror_lens_mirror_cavity_analysis(cavity=cavity_completed)
plt.show()

