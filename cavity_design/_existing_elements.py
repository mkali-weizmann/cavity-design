import numpy as np

from ._surfaces import generate_aspheric_lens_params, CurvedMirror
from ._cavity import (
    OpticalSystem,
    CurvedRefractiveSurface,
    CurvatureSigns,
    AsphericRefractiveSurface,
    FlatRefractiveSurface,
)
from ._utils import PHYSICAL_SIZES_DICT, LEFT, ORIGIN, RIGHT, INCH

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

eksma_lens_params = generate_aspheric_lens_params(
    back_focal_length=17.001e-3,
    T_c=4.35e-3,
    forward_normal=LEFT,
    flat_faces_center=ORIGIN + 15e-3 * LEFT,
    n=PHYSICAL_SIZES_DICT["material_properties_fused_silica"].refractive_index,
    diameter=INCH / 2,
    polynomial_degree=10,
    n_outside=1,
    name="Eksma 20mm",
)
EKSMA_LENS_20mm_ASPHERIC = OpticalSystem.from_params(
    params=eksma_lens_params, use_paraxial_ray_tracing=False, p_is_trivial=True, t_is_trivial=True
)
del eksma_lens_params
