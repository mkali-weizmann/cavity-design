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
)

EKSMA_LENS_20mm_ASPHERIC = OpticalSystem(
    elements=generate_aspheric_lens(
        back_focal_length=17.001e-3,
        T_c=4.35e-3,
        forward_normal=LEFT,
        flat_faces_center=ORIGIN + 15e-3 * LEFT,
        n=PHYSICAL_SIZES_DICT["material_properties_fused_silica"].refractive_index,
        diameter=INCH / 2,
        polynomial_degree=10,
        n_outside=1,
        name="Eksma 20mm",
    ),
    use_paraxial_ray_tracing=False,
    p_is_trivial=True,
    t_is_trivial=True,
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
                [0.00000000e+00, 1.07802206e+02, 5.72281050e+05, 4.21121557e+09, 3.16313244e+13,
                 -6.49022844e+17, 2.19949315e+18, 5.98323954e+21, 1.68309599e+25, 4.85597863e+28,
                 1.42904143e+32]),
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
)

EDMUND_8MM_ASPHERIC_31074 = OpticalSystem(elements=[
        AsphericRefractiveSurface(
            center=ORIGIN,
            outwards_normal=LEFT,
            # Note: the coefficients must not be negated (coef[1] must be positive) — the curvature direction is
            # encoded in the outwards normal (LEFT here), as the AsphericSurface assertion requires.
            polynomial_coefficients=np.array([ 0.00000000e+00,  1.07802206e+02,  5.72279797e+05,  4.21121124e+09, 3.16313103e+13, -6.49022889e+17,  2.19934550e+18,  5.98275755e+21, 1.68293781e+25,  4.85545706e+28,  1.42886875e+32]),
            n_1=1,
            n_2=1.574,
            name="Edmund 8mm aspheric - convex side",
            diameter=6.3e-03,
            curvature_sign=CurvatureSigns.convex,
        ),
    FlatRefractiveSurface(
        center=3.43 * RIGHT,
        outwards_normal=RIGHT,
        n_1=1.574,
        n_2=1,
        name="Edmund 8mm aspheric - flat side",
        diameter=6.3e-3
    )
], use_paraxial_ray_tracing=False)

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
    ("EKSMA_LENS_20mm_ASPHERIC", EKSMA_LENS_20mm_ASPHERIC),
    ("THORLABS_8MM_ASPHERIC", THORLABS_8MM_ASPHERIC),
    ("EDMUND_8MM_ASPHERIC_31074", EDMUND_8MM_ASPHERIC_31074),
]:
    register_existing_element(_catalog_name, _element)
del _catalog_name, _element