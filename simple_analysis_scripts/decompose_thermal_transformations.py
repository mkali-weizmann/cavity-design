from cavity import *
np.seterr(all="raise")

params = [
          OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'                  , x=-4.999954683912563e-03  , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'                     , x=6.387599281689135e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=-5.488e-03              , curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'                  , x=4.074677357638641e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2e-01                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=2.54e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  ), polynomial_coefficients=None)]
cavity = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=LAMBDA_0_LASER,
                            set_central_line=True,
                            set_mode_parameters=True,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4,
                            use_paraxial_ray_tracing=True,
                            debug_printing_level=1,
                            )

results_dict, cavities = cavity.analyze_thermal_transformation(arm_index_for_NA=0)
# %%

def mm_format(value, tick_number):
    return f"{value * 1e3:.2f}"


def cm_format(value, tick_number):
    return f"{value * 1e2:.2f}"
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
for i, key_trans in enumerate(results_dict.keys()):
    ax = axes.flatten()[i]
    cavities[i].plot(ax=ax)
    ax.set_title(f"{key_trans}: Power -> NA *= {results_dict[key_trans]:.2f}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(mm_format))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(cm_format))
    ax.set_ylabel("y [mm]")
    ax.set_xlabel("x [cm]")
fig.suptitle(generate_mirror_lens_mirror_cavity_textual_summary(cavity=cavity))
fig.tight_layout()
plt.savefig(f"outputs/figures/thermal-transformations/mirror-lens-mirror.svg")
plt.show()