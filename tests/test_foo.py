import numpy as np
from cavity import (
    CurvedMirror,
    CurvatureSigns,
    Cavity,
    LAMBDA_0_LASER,
)

def test_fabry_perot_mode_finding():
    # Compares the numerical result from the analytical solution for a simple Fabry-Perot cavity
    R_1 = 5e-3
    R_2 = 5e-3
    u = 1e-5
    L = R_1 + R_2 - u
    surface_1 = CurvedMirror(center=np.array([0, 0, -R_1]),
                                    outwards_normal=np.array([0, 0, -1]),
                                    radius=R_1,
                                    curvature_sign=CurvatureSigns.concave,
                                    diameter=0.01)
    surface_2 = CurvedMirror(center=np.array([0, 0, -R_1 + L]),
                                    outwards_normal=np.array([0, 0, 1]),
                                    radius=R_2,
                                    curvature_sign=CurvatureSigns.concave,
                                    diameter=0.01)
    cavity = Cavity(physical_surfaces=[surface_1, surface_2],
                    standing_wave=True,
                    lambda_0_laser=LAMBDA_0_LASER,
                    power=1e3,
                    use_paraxial_ray_tracing=False)
    theoretical_reighly_range = np.sqrt(u * L) / 2
    actual_reighly_range = cavity.arms[0].mode_parameters.z_R[0]

    theoretical_waist = np.sqrt(LAMBDA_0_LASER * theoretical_reighly_range / np.pi)
    actual_waist = cavity.arms[0].mode_parameters.w_0[0]
    # print(f'Theoretical Reighly range: {theoretical_reighly_range}, Actual Reighly range: {actual_reighly_range}')
    # print(f'Theoretical Waist: {theoretical_waist}, Actual Waist: {actual_waist}')
    assert (theoretical_reighly_range / actual_reighly_range - 1) < 1e-6, f'Reighly range mismatch: theoretical {theoretical_reighly_range}, actual {actual_reighly_range}'
    assert (theoretical_waist / actual_waist - 1) < 1e-6, f'Waist mismatch: theoretical {theoretical_waist}, actual {actual_waist}'
