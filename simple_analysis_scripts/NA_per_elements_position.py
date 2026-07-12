from matplotlib import use
use("TkAgg")

from cavity_design import *
from tqdm import tqdm


# No lens:
# params = [
#           OpticalSurfaceParams(name='Laser Optik Mirror'     ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=5e-03                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None), [
#           OpticalSurfaceParams(name='low curvature side - Edmund 4.03mm spherical version',surface_type='curved_refractive_surface'      , x=2.452564065600806e-03   , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=1.267523034472214e-02   , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.574e+00               , n_outside_or_before=1e+00                   , diameter=5.1e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
#           OpticalSurfaceParams(name='high curvature side - Edmund 4.03mm spherical version',surface_type='curved_refractive_surface'      , x=5.552564065600805e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.619751468026235e-03   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.574e+00               , diameter=5.1e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)],
#           OpticalSurfaceParams(name='End mirror'             ,surface_type='curved_mirror'                  , x=3.056414817693546e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2e-01                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=np.nan                  , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)]


# 10cm spherical lens
params = [
          OpticalSurfaceParams(name='Laser Optik Mirror'     ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=5e-03                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None), [
          OpticalSurfaceParams(name='Edmund 4.03 aspheric - low ROC side',surface_type='aspheric_surface'               , x=2.321734075962478e-03   , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=1.267523033789814e-02   , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.574e+00               , n_outside_or_before=1e+00                   , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=np.array([-0.00000000e+00,  3.94470149e+01, -4.40171309e+06,  9.01563757e+10, 1.60183137e+15,  3.23783960e+15,  1.51148859e+19,  7.39192926e+22, 3.73825954e+26,  1.93899408e+30,  1.02584959e+34])),
          OpticalSurfaceParams(name='Edmund 4.03 aspheric - high ROC side',surface_type='aspheric_surface'               , x=5.421734075962479e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.619751472665783e-03   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.574e+00               , diameter=6.3e-03                 , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=np.array([0.00000000e+00, 1.90857799e+02, 3.33213215e+06, 1.08218014e+11,  4.32910754e+15, 6.27836273e+20, 2.70839583e+25, 1.72102573e+30,  1.13089247e+35, 7.62167883e+39, 5.23938302e+44]))
         ], [
          OpticalSurfaceParams(name='Thorlabs 100mm plano convex - convex',surface_type='curved_refractive_surface'      , x=2.042173407596248e-02   , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=5.15e-02                , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.507e+00               , n_outside_or_before=1e+00                   , diameter=2.54e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='Thorlabs 100mm plano convex - left',surface_type='flat_refractive_surface'        , x=2.402173407596248e-02   , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=0                       , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.507e+00               , diameter=2.54e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)],
          OpticalSurfaceParams(name='End mirror'             ,surface_type='curved_mirror'                  , x=3.173008772482745e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2e-01                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=2.54e-02                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None)]

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
plt.title(f"NA_short = {cavity.mode_parameters[0].NA[0]:.2e}, NA_long = {cavity.mode_parameters[cavity.last_arm_index].NA[0]:.2e}\nlength short = {cavity.central_line[0].length*1000:.2f}mm, length long = {cavity.central_line[cavity.last_arm_index].length*1000:.2f}mm")
plt.show()

# %% 2d map:
def equality_equation(x, coef):
    quad_deriv = 2 * coef[1] * x
    higher_deriv = sum(2 * n * coef[n] * x ** (2 * n - 1) for n in range(2, len(coef)))
    return abs(quad_deriv) - abs(higher_deriv)

cavity = Cavity.from_params(params=params, standing_wave=True,
                                    lambda_0_laser=LAMBDA_0_LASER, power=28e3, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=False, set_central_line=True, set_mode_parameters=True)
collimation_point = cavity[0].radius + back_focal_length_of_lens_object(lens_object=cavity[1])
long_arm_lengths = np.array([29e-2, 30e-2, 31e-2, 32e-2, 33e-2, 34e-2, 37e-2])
short_arm_lengths = np.linspace(collimation_point-2e-4, collimation_point+1e-4, 300)
NAs = np.zeros(shape=(len(long_arm_lengths), len(short_arm_lengths)))
mode_spacings = np.full(shape=(len(long_arm_lengths), len(short_arm_lengths)), fill_value=np.nan)
zero_derivative_points = np.full(shape=(len(long_arm_lengths), len(short_arm_lengths)), fill_value=np.nan)
quadratic_order_zero_derivative_point = np.full(shape=(len(long_arm_lengths), len(short_arm_lengths)), fill_value=np.nan)
quadratic_quadratic_equality_point = np.full(shape=(len(long_arm_lengths), len(short_arm_lengths)), fill_value=np.nan)
focii_to_lens = np.zeros(shape=(len(long_arm_lengths)))
for i, long_arm_length in enumerate(long_arm_lengths):  #
    cavity.place_elements(elements=cavity[-1], position=long_arm_length * RIGHT, reference_center=cavity.surfaces[-2], recalculate_optic=False)
    flag=False
    for j, short_arm_length in enumerate(short_arm_lengths):
        cavity.place_elements(elements=cavity[1], position = short_arm_length * RIGHT, reference_center=cavity[0])
        NAs[i, j] = cavity.arms[0].mode_parameters.NA[0]
        mode_spacings[i, j] = cavity.mode_spacing_transversal_apparent
        if not np.isnan(cavity.arms[0].mode_parameters.NA[0]):
            results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=50,
                                                          phi_max=np.arcsin(0.3), print_tests=False)
            potential_polynomial = results_dict["polynomial_residuals_mirror"]
            quadratic_quadratic_equality_point[i, j] = np.sqrt(np.abs(potential_polynomial.coef[1] / potential_polynomial.coef[2]))
            # zero_derivative_points[i, j] = results_dict["zero_derivative_point"]
            # quadratic_order_zero_derivative_point[i, j] = np.sqrt(np.abs(potential_polynomial.coef[1] / (2 * potential_polynomial.coef[2])))
        if j == 0:
            focii_to_lens[i] = cavity.surfaces[-1].center[0] - cavity.surfaces[2].center[0] - cavity.surfaces[-1].radius
        if cavity.arms[0].mode_parameters.NA[0] > 0.07 and flag is False:
            focii_to_lens[i] = cavity.surfaces[-1].center[0] - cavity.surfaces[2].center[0] - cavity.surfaces[-1].radius
            flag=True

# %%
plot_different_axes = True
if plot_different_axes:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
else:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax2 = ax.twinx()
    # ax.set_zorder(ax2.get_zorder() + 1)  # Make ax "on top" for coordinate display

for i in range(len(long_arm_lengths)):
    na_line, = ax.plot(short_arm_lengths * 1e3, NAs[i, :], label=f"Mirror-to-spherical = {long_arm_lengths[i]*100:.2f}cm")
    ax.plot(short_arm_lengths * 1e3, zero_derivative_points[i, :], color=na_line.get_color(), linestyle='-.', label=f"maximally allowed NA")
    # ax.plot(short_arm_lengths * 1e3, quadratic_order_zero_derivative_point[i, :], color=na_line.get_color(), linestyle=':')
    if i == 0:
        ax.plot(short_arm_lengths * 1e3, quadratic_quadratic_equality_point[i, :], color=na_line.get_color(), linestyle=':', alpha=0.3, label="Quadratic-quartic equality")
    else:
        ax.plot(short_arm_lengths * 1e3, quadratic_quadratic_equality_point[i, :], color=na_line.get_color(),
                linestyle=':', alpha=0.3)
    ax2.plot(short_arm_lengths * 1e3, mode_spacings[i, :] / 1e6, linestyle='--', label=f"Mirror-to-spherical = {long_arm_lengths[i]*100:.2f}cm")
ax2.set_ylim(0, 300)
ax2.set_ylabel("Mode Spacing [MHz]")
ax.set_ylabel('Short Arm Numerical Aperture')
ax.axvline(collimation_point * 1e3, color='k', linestyle='--', linewidth=1, label='Collimation point')
ax.set_ylim(0, 0.32)
ax.grid()
if plot_different_axes:
    ax2.set_xlabel('Short Arm Length (mm)')
    ax2.axvline(collimation_point * 1e3, color='k', linestyle='--', linewidth=1)
    ax2.grid()
    fig.subplots_adjust(right=0.68)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    ax.set_title(f"spherical focal length = {focal_length_of_lens_object(cavity[2]) * 1000:.0f} mm")
else:
    ax.set_xlabel('Short Arm Length (mm)')
    fig.subplots_adjust(right=0.68)
    ax.legend(loc='center left', bbox_to_anchor=(1.12, 0.5), borderaxespad=0.0)
    plt.title(f"spherical focal length = {focal_length_of_lens_object(cavity[2]) * 1000:.0f} mm")
# obsidian_path=get_obsidian_save_path(filename='NA as a function of mirrors.svg')
# plt.savefig(obsidian_path)
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
# short_arm_lengths = np.array([7.35e-3, 7.45e-3, 7.55e-3, 7.65e-3])
# NAs = np.linspace(0.03, 0.15, 100)
# long_arm_lengths = np.full(shape=(len(short_arm_lengths), len(NAs)), fill_value=np.nan)
# mode_spacings = np.full(shape=(len(short_arm_lengths), len(NAs)), fill_value=np.nan)
# for i, short_arm_length in enumerate(short_arm_lengths):
#     optical_system = OpticalSystem.from_params(params=params[0:-1], use_paraxial_ray_tracing=True, lambda_0_laser=LAMBDA_0_LASER, p_is_trivial=True, t_is_trivial=True,)
#     optical_system.place_elements(elements=optical_system[1], position=short_arm_length * RIGHT, reference_center=optical_system[0])
#     for j, NA in enumerate(NAs):
#         try:
#             cavity=optical_system.complete_to_cavity(NA=NA, end_mirror_ROC=2e-1)
#             long_arm_lengths[i, j] = cavity.surfaces[-1].center[0] - cavity.surfaces[2].center[0]
#             mode_spacings[i, j] = cavity.mode_spacing_transversal_apparent
#         except ValueError:
#             continue
#
#
# plt.close('all')
# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax2 = ax1.twinx()
# for i in range(len(short_arm_lengths)):
#     ax1.plot(long_arm_lengths[i, :]*100, NAs, label=f"Short arm={short_arm_lengths[i]*1e3:.2f}mm - NA")
#     ax2.plot(long_arm_lengths[i, :]*100, mode_spacings[i, :] / 1e6, linestyle='--', label=f"Short arm={short_arm_lengths[i]*1e3:.2f}mm - df")
# ax1.legend()
# ax1.grid()
# ax1.set_xlabel("Large mirror to aspheric distance [cm]")
# ax1.set_ylabel("NA")
# ax2.set_ylabel("Mode spacing [MHz]")
# plt.tight_layout()
# plt.show()
#
# fig, ax = plt.subplots(figsize=(10, 6))
# for i in range(len(short_arm_lengths)):
#     ax.plot(NAs, mode_spacings[i, :] / 1e6, label=f"Short arm={short_arm_lengths[i]*1e3:.2f}mm")
# ax.set_xlabel("NA")
# ax.set_ylabel("Mode spacing [MHz]")
# ax.legend()
# ax.grid()
# plt.tight_layout()
# plt.show()


# %% Long arm perturbation
# perturbations_large_mirror = np.linspace(-4e-2, 4e-2, 100)
#
# NAs = np.zeros_like(perturbations_large_mirror)
# long_arm_lengths = np.zeros_like(perturbations_large_mirror)
# for i, perturbation_value in enumerate(perturbations_large_mirror):
#     perturbation_pointer = PerturbationPointer(element_index=2, parameter_name=ParamsNames.x, perturbation_value=perturbation_value)
#     perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointer)
#     NAs[i] = perturbed_cavity.arms[0].mode_parameters.NA[0]
#     long_arm_lengths[i] = perturbed_cavity.arms[2].central_line.length
#
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(long_arm_lengths * 1e2, NAs, marker='o', markersize=2)
# ax.set_xlabel('Long arm length (cm)')
# ax.set_ylabel('Short Arm Numerical Aperture')
# ax.set_title('Short Arm Numerical Aperture as a Function of Long Arm Length')
# ax.grid()
# plt.tight_layout()
# # plt.savefig(r'figures\NA_as_a_function_of_small_mirror')
# plt.show()

# %% Short arm perturbation
# perturbations_aspheric_lens = np.linspace(-5e-5, 2e-4, 100)
#
# NAs = np.zeros_like(perturbations_aspheric_lens)
# short_arm_lengths = np.zeros_like(perturbations_aspheric_lens)
# for i, perturbation_value in enumerate(perturbations_aspheric_lens):
#     perturbation_pointer = PerturbationPointer(element_index=0, parameter_name=ParamsNames.x, perturbation_value=perturbation_value)
#     perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointer)
#     NAs[i] = perturbed_cavity.arms[0].mode_parameters.NA[0]
#     short_arm_lengths[i] = perturbed_cavity.arms[0].central_line.length
#
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(short_arm_lengths * 1e2, NAs, marker='o', markersize=2)
# ax.set_xlabel('Short arm length (cm)')
# ax.set_ylabel('Short Arm Numerical Aperture')
# ax.set_title('Short Arm Numerical Aperture as a Function of Small arm length')
# ax.grid()
# plt.tight_layout()
# # plt.savefig(r'figures\NA_as_a_function_of_small_mirror')
# plt.show()