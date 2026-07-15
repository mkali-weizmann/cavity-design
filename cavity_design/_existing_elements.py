import numpy as np

from ._surfaces import generate_aspheric_lens, CurvedMirror
from ._cavity import (
    OpticalSystem,
    CurvedRefractiveSurface,
    CurvatureSigns,
    AsphericRefractiveSurface,
    FlatRefractiveSurface,
    register_existing_element,
)
from ._utils import PHYSICAL_SIZES_DICT, LEFT, ORIGIN, RIGHT, INCH, MaterialProperties

LASER_OPTIK_MIRROR = CurvedMirror(
    radius=5e-3,
    origin=ORIGIN,
    diameter=7.75e-3,
    outwards_normal=LEFT,
    name="Laser Optik Mirror",
    material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
)
COASTLINE_20CM_MIRROR = CurvedMirror(
    radius=20e-2,
    diameter=INCH,
    outwards_normal=RIGHT,
    name="Coastline 20cm Mirror",
    material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
)
COASTLINE_50CM_MIRROR = CurvedMirror(
    radius=50e-2,
    diameter=INCH,
    outwards_normal=RIGHT,
    name="Coastline 50cm Mirror",
    material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
)

LASER_OPTIK_MIRROR_REFRACTIVE = OpticalSystem(
    elements=[
        CurvedRefractiveSurface(
            radius=5e-3,
            diameter=7.75e-3,
            outwards_normal=LEFT,
            origin=ORIGIN,
            name="Laser Optik Mirror - Concave",
            n_1=1,
            curvature_sign=CurvatureSigns.concave,
            n_2=PHYSICAL_SIZES_DICT["material_properties_fused_silica"].refractive_index,
            material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
        ),
        CurvedRefractiveSurface(
            curvature_sign=CurvatureSigns.concave,
            radius=5e-3,
            diameter=7.75e-3,
            outwards_normal=LEFT,
            origin=ORIGIN + 3.45e-3 * LEFT,
            name="Laser Optik Mirror - Convex",
            n_1=PHYSICAL_SIZES_DICT["material_properties_fused_silica"].refractive_index,
            n_2=1,
            material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
        ),
    ],
    use_paraxial_ray_tracing=True,
    name='LaserOptik Mirror - Refractive Version'
)

THORLABS_35MM_COLLIMATING_LENS = OpticalSystem(
    elements=[
        CurvedRefractiveSurface(
            name="Thorlabs 35mm biconvex - right",
            radius=34.9e-3,
            outwards_normal=RIGHT,
            diameter=INCH,
            curvature_sign=CurvatureSigns.convex,
            n_1=1,
            n_2=PHYSICAL_SIZES_DICT["material_properties_bk7"].refractive_index,
        ),
        CurvedRefractiveSurface(
            name="Thorlabs 35mm biconvex - left",
            radius=34.9e-3,
            outwards_normal=LEFT,
            center=(LASER_OPTIK_MIRROR_REFRACTIVE.surfaces[1].center + 6.8e-3 * LEFT) * 1j,
            diameter=INCH,
            curvature_sign=CurvatureSigns.concave,
            n_1=PHYSICAL_SIZES_DICT["material_properties_bk7"].refractive_index,
            n_2=1,
        ),
    ],
    use_paraxial_ray_tracing=True,
    name='Thorlabs 35mm Collimating Lens'
)

EDMUND_4p03MM_ASPHERIC_SPHERICAL_VERSION = OpticalSystem(
    elements=[
        CurvedRefractiveSurface(
            center=ORIGIN,
            name="low curvature side - Edmund 4.03mm spherical version",
            radius=(1 / 7.889402975752558833e-02) * 1e-3,
            outwards_normal=LEFT,
            diameter=5.1e-3,
            curvature_sign=CurvatureSigns.convex,
            n_1=1,
            n_2=1.574,
        ),
        CurvedRefractiveSurface(
            name="high curvature side - Edmund 4.03mm spherical version",
            radius=(1 / 3.817155986760137343e-01) * 1e-3,
            outwards_normal=RIGHT,
            center=3.1e-3 * RIGHT,
            diameter=5.1e-3,
            curvature_sign=CurvatureSigns.concave,
            n_1=1.574,
            n_2=1,
        ),
    ],
    use_paraxial_ray_tracing=True,
    name='Edmund 4.03mm Aspheric Spherical Version',
)

EDMUND_4p03MM_ASPHERIC = OpticalSystem(
    elements=[
        AsphericRefractiveSurface(
            center=ORIGIN,
            outwards_normal=LEFT,
            polynomial_coefficients=-np.array(
                [
                    0.00000000e00,
                    -3.94470149e01,
                    4.40171309e06,
                    -9.01563757e10,
                    -1.60183137e15,
                    -3.23783960e15,
                    -1.51148859e19,
                    -7.39192926e22,
                    -3.73825954e26,
                    -1.93899408e30,
                    -1.02584959e34,
                ]
            ),
            n_1=1,
            n_2=1.574,
            name="Edmund 4.03 aspheric - low ROC side",
            diameter=6.3e-03,
            curvature_sign=CurvatureSigns.convex,
        ),
        AsphericRefractiveSurface(
            center=3.1e-3 * RIGHT,
            outwards_normal=RIGHT,
            polynomial_coefficients=np.array(
                [
                    0.00000000e00,
                    1.90857799e02,
                    3.33213215e06,
                    1.08218014e11,
                    4.32910754e15,
                    6.27836273e20,
                    2.70839583e25,
                    1.72102573e30,
                    1.13089247e35,
                    7.62167883e39,
                    5.23938302e44,
                ]
            ),
            n_1=1.574,
            n_2=1,
            name="Edmund 4.03 aspheric - high ROC side",
            diameter=6.3e-03,
            curvature_sign=CurvatureSigns.concave,
        ),
    ],
    use_paraxial_ray_tracing=False,
    p_is_trivial=True,
    t_is_trivial=True,
    name='Edmund 4.03mm Aspheric'
)

THOLABS_100MM_PLANO_CONVEX_LENS = OpticalSystem(
    elements=[
        CurvedRefractiveSurface(
            name="Thorlabs 100mm plano convex - convex",
            radius=51.5e-3,
            outwards_normal=LEFT,
            diameter=INCH,
            curvature_sign=CurvatureSigns.convex,
            n_1=1,
            n_2=PHYSICAL_SIZES_DICT["material_properties_bk7"].refractive_index,
        ),
        FlatRefractiveSurface(
            name="Thorlabs 100mm plano convex - left",
            center=ORIGIN + 3.6e-3 * RIGHT * 1j,
            outwards_normal=LEFT,
            diameter=INCH,
            n_1=PHYSICAL_SIZES_DICT["material_properties_bk7"].refractive_index,
            n_2=1,
        ),
    ],
    use_paraxial_ray_tracing=True,
    p_is_trivial=True,
    t_is_trivial=True,
    name='Thorlabs 100mm Plano Convex Lens'
)

DUMMY_LENS = OpticalSystem(
    elements=[
        CurvedRefractiveSurface(
            name="Dummy lens - convex side",
            radius=102e-3,
            outwards_normal=LEFT,
            diameter=INCH,
            curvature_sign=CurvatureSigns.convex,
            n_1=1,
            n_2=PHYSICAL_SIZES_DICT["material_properties_bk7"].refractive_index,
        ),
        FlatRefractiveSurface(
            name="Dummy lens - flat side",
            center=ORIGIN + 3.6e-3 * RIGHT * 1j,
            outwards_normal=LEFT,
            diameter=INCH,
            n_1=PHYSICAL_SIZES_DICT["material_properties_bk7"].refractive_index,
            n_2=1,
        ),
    ],
    use_paraxial_ray_tracing=True,
    p_is_trivial=True,
    t_is_trivial=True,
    name='Dummy Lens'
)


EKSMA_LENS_20MM_ASPHERIC = OpticalSystem(
elements=[
        FlatRefractiveSurface(
            name="Eksma 20mm aspheric - flat side",
            center=ORIGIN,
            outwards_normal=LEFT,
            n_1=1,
            n_2=1.4496,
            diameter=6.35e-3,
            thermal_properties=MaterialProperties(),
        ),
        AsphericRefractiveSurface(
            name="Eksma 20mm aspheric - convex side",
            center=3.434e-3 * RIGHT,
            outwards_normal=RIGHT,
            polynomial_coefficients=np.array([0.00000000e+00, 5.51085639e+01, 7.27331328e+04, 1.61094176e+08,
       3.91471107e+11, 7.78315423e+14, 2.34007091e+18, 7.37064994e+21,
       2.40071869e+25, 8.01995895e+28, 2.73277068e+32]),
            n_1=1.4496,
            n_2=1,
            curvature_sign=CurvatureSigns.concave,
            diameter=6.35e-3,
            material_properties=MaterialProperties(),
        ),
    ],
    use_paraxial_ray_tracing=False,
    p_is_trivial=True,
    t_is_trivial=True,
    name='Eksma 20mm aspheric',
)

THORLABS_8MM_ASPHERIC = OpticalSystem(
    elements=[
        FlatRefractiveSurface(
            name="aspheric_lens_flat",
            center=ORIGIN,
            outwards_normal=LEFT,
            n_1=1,
            n_2=1.577,
            diameter=8.2e-3,
            thermal_properties=MaterialProperties(),
        ),
        AsphericRefractiveSurface(
            name="aspheric_lens_convex",
            center=3.434e-3 * RIGHT,
            outwards_normal=RIGHT,
            polynomial_coefficients=np.array(
                [
                    0.00000000e00,
                    1.07802206e02,
                    5.72281050e05,
                    4.21121557e09,
                    3.16313244e13,
                    -6.49022844e17,
                    2.19949315e18,
                    5.98323954e21,
                    1.68309599e25,
                    4.85597863e28,
                    1.42904143e32,
                ]
            ),
            n_1=1.577,
            n_2=1,
            curvature_sign=CurvatureSigns.concave,
            diameter=8.2e-3,
            material_properties=MaterialProperties(),
        ),
    ],
    use_paraxial_ray_tracing=False,
    p_is_trivial=True,
    t_is_trivial=True,
    name='Thorlabs 8mm Aspheric',
)

EDMUND_8MM_ASPHERIC_31074 = OpticalSystem(
    elements=[
    FlatRefractiveSurface(
            center=ORIGIN,
            outwards_normal=LEFT,
            n_1=1,
            n_2=1.574,
            name="Edmund 8mm aspheric - flat side",
            diameter=6.3e-3,
        ),
        AsphericRefractiveSurface(
            center=3.43e-3 * RIGHT,
            outwards_normal=RIGHT,
            # Note: the coefficients must not be negated (coef[1] must be positive) — the curvature direction is
            # encoded in the outwards normal (LEFT here), as the AsphericSurface assertion requires.
            polynomial_coefficients=np.array(
                [
                    0.00000000e00,
                    1.07802206e02,
                    5.72279797e05,
                    4.21121124e09,
                    3.16313103e13,
                    -6.49022889e17,
                    2.19934550e18,
                    5.98275755e21,
                    1.68293781e25,
                    4.85545706e28,
                    1.42886875e32,
                ]
            ),
            n_1=1.574,
            n_2=1,
            name="Edmund 8mm aspheric - convex side",
            diameter=6.3e-03,
            curvature_sign=CurvatureSigns.concave,
        ),

    ],
    use_paraxial_ray_tracing=False,
    name='Edmund 8mm Aspheric - 31074',
)

EDMUND_6MM_ASPHERIC_87127 = OpticalSystem(
    elements=[
    FlatRefractiveSurface(
            center=ORIGIN,
            outwards_normal=LEFT,
            n_1=1,
            n_2=1.784,
            name="Edmund 6mm aspheric - flat side",
            diameter=6.3e-3,
        ),
        AsphericRefractiveSurface(
            center=5.16e-3 * RIGHT,
            outwards_normal=RIGHT,
            # Note: the coefficients must not be negated (coef[1] must be positive) — the curvature direction is
            # encoded in the outwards normal (LEFT here), as the AsphericSurface assertion requires.
            polynomial_coefficients=np.array([ 0.00000000e+00,  1.01043475e+02,  4.80138198e+05,  1.65064522e+09,
       -6.99279234e+13,  2.84292367e+15, -1.01547369e+19,  3.79992950e+22,
       -1.47041906e+26,  5.83582271e+29, -2.36245328e+33]),
            n_1=1.784,
            n_2=1,
            name="Edmund 6mm aspheric - convex side",
            diameter=6.3e-03,
            curvature_sign=CurvatureSigns.concave,
        ),

    ],
    use_paraxial_ray_tracing=False,
    name='Edmund 6mm Aspheric - 87127',
)



# Register every catalog element: each gets tagged with its variable name (the tag survives deepcopy, so placed and
# moved copies stay recognizable) and a pristine copy is stored, letting OpticalSystem.init_syntax render these
# elements compactly as e.g. EKSMA_LENS_20mm_ASPHERIC.to_position(...).
for _catalog_name, _element in [
    ("LASER_OPTIK_MIRROR", LASER_OPTIK_MIRROR),
    ("COASTLINE_20CM_MIRROR", COASTLINE_20CM_MIRROR),
    ("COASTLINE_50CM_MIRROR", COASTLINE_50CM_MIRROR),
    ("LASER_OPTIK_MIRROR_REFRACTIVE", LASER_OPTIK_MIRROR_REFRACTIVE),
    ("THORLABS_35MM_COLLIMATING_LENS", THORLABS_35MM_COLLIMATING_LENS),
    ("EDMUND_4p03MM_ASPHERIC_SPHERICAL_VERSION", EDMUND_4p03MM_ASPHERIC_SPHERICAL_VERSION),
    ("EDMUND_4p03MM_ASPHERIC", EDMUND_4p03MM_ASPHERIC),
    ("THOLABS_100MM_PLANO_CONVEX_LENS", THOLABS_100MM_PLANO_CONVEX_LENS),
    ("EKSMA_LENS_20mm_ASPHERIC", EKSMA_LENS_20MM_ASPHERIC),
    ("THORLABS_8MM_ASPHERIC", THORLABS_8MM_ASPHERIC),
    ("EDMUND_8MM_ASPHERIC_31074", EDMUND_8MM_ASPHERIC_31074),
    ("EDMUND_6MM_ASPHERIC_87127", EDMUND_6MM_ASPHERIC_87127),
    ("DUMMY_LENS", DUMMY_LENS)

]:
    register_existing_element(_catalog_name, _element)
del _catalog_name, _element
