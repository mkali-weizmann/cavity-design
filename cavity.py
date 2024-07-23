# %%
from __future__ import annotations

import warnings

from utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from dataclasses import dataclass
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from datetime import datetime
from hashlib import md5
from tqdm import tqdm

pd.set_option("display.max_rows", 500)
pd.options.display.float_format = "{:.3e}".format

np.seterr(all="raise")


@dataclass
class MaterialProperties:
    refractive_index: Optional[float] = None
    alpha_expansion: Optional[float] = None
    beta_surface_absorption: Optional[float] = None
    kappa_conductivity: Optional[float] = None
    dn_dT: Optional[float] = None
    nu_poisson_ratio: Optional[float] = None
    alpha_volume_absorption: Optional[float] = None
    intensity_reflectivity: Optional[float] = None
    intensity_transmittance: Optional[float] = None
    temperature: Optional[float] = np.nan

    def __repr__(self):
        return (
            f"MaterialProperties("
            f"refractive_index={pretty_print_number(self.refractive_index)}, "
            f"alpha_expansion={pretty_print_number(self.alpha_expansion)}, "
            f"beta_surface_absorption={pretty_print_number(self.beta_surface_absorption)}, "
            f"kappa_conductivity={pretty_print_number(self.kappa_conductivity)}, "
            f"dn_dT={pretty_print_number(self.dn_dT)}, "
            f"nu_poisson_ratio={pretty_print_number(self.nu_poisson_ratio)}, "
            f"alpha_volume_absorption={pretty_print_number(self.alpha_volume_absorption)}, "
            f"intensity_reflectivity={pretty_print_number(self.intensity_reflectivity)}, "
            f"intensity_transmittance={pretty_print_number(self.intensity_transmittance)}, "
            f"temperature={pretty_print_number(self.temperature)})"
        )

    @property
    def to_array(self) -> np.ndarray:
        return np.array(
            [
                nvl(self.refractive_index),
                nvl(self.alpha_expansion, np.nan),
                nvl(self.beta_surface_absorption, np.nan),
                nvl(self.kappa_conductivity, np.nan),
                nvl(self.dn_dT, np.nan),
                nvl(self.nu_poisson_ratio, np.nan),
                nvl(self.alpha_volume_absorption),
                nvl(self.intensity_reflectivity),
                nvl(self.intensity_transmittance),
                nvl(self.temperature, np.nan),
            ]
        )  # Note: When changing the order or adding this list - the order or added items should be also updated in the from_array method of OpticalElementParams


PHYSICAL_SIZES_DICT = {
    "thermal_properties_sapphire": MaterialProperties(
        refractive_index=1.76,
        alpha_expansion=5.5e-6,  # https://www.shinkosha.com/english/techinfo/feature/thermal-properties-of-sapphire/#:~:text=Sapphire%20has%20a%20large%20linear,very%20resistant%20to%20thermal%20shock., https://www.roditi.com/SingleCrystal/Sapphire/Properties.html
        beta_surface_absorption=1e-6,  # DUMMY
        kappa_conductivity=46.06,  # https://www.google.com/search?q=sapphire+thermal+conductivity&rlz=1C1GCEB_enIL1023IL1023&oq=sapphire+thermal+c&aqs=chrome.0.35i39i650j69i57j0i20i263i512j0i22i30l3j0i10i15i22i30j0i22i30l3.3822j0j1&sourceid=chrome&ie=UTF-8, https://www.shinkosha.com/english/techinfo/feature/thermal-properties-of-sapphire/, https://www.shinkosha.com/english/techinfo/feature/thermal-properties-of-sapphire/
        dn_dT=11.7e-6,  # https://secwww.jhuapl.edu/techdigest/Content/techdigest/pdf/V14-N01/14-01-Lange.pdf
        nu_poisson_ratio=0.3,  # https://www.google.com/search?q=sapphire+poisson+ratio&rlz=1C1GCEB_enIL1023IL1023&sxsrf=AB5stBgEUZwh7l9RzN9GwxjMPCw_DcShAw%3A1688647440018&ei=ELemZI1h0-2SBaukk-AH&ved=0ahUKEwiNqcD2jfr_AhXTtqQKHSvSBHwQ4dUDCA8&uact=5&oq=sapphire+poisson+ratio&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIECAAQHjIICAAQigUQhgMyCAgAEIoFEIYDMggIABCKBRCGAzIICAAQigUQhgMyCAgAEIoFEIYDOgoIABBHENYEELADSgQIQRgAUJsFWJsFYNQJaAFwAXgAgAF5iAF5kgEDMC4xmAEAoAEBwAEByAEI&sclient=gws-wiz-serp
        alpha_volume_absorption=100e-6
        * 100,  # The data is in ppm/cm and I convert it to ppm/m, hence the "*100".  # https://labcit.ligo.caltech.edu/~ligo2/pdf/Gustafson2c.pdf  # https://www.nature.com/articles/s41598-020-80313-1  # https://www.crystran.co.uk/optical-materials/sapphire-al2o3,
        intensity_reflectivity=100e-6,  # DUMMY - for lenses
        intensity_transmittance=1 - 100e-6 - 1e-6,
    ),  # DUMMY - for lenses
    "thermal_properties_ULE": MaterialProperties(
        alpha_expansion=7.5e-8,  # https://en.wikipedia.org/wiki/Ultra_low_expansion_glass#:~:text=It%20has%20a%20thermal%20conductivity,C%20%5B1832%20%C2%B0F%5D, https://www.corning.com/media/worldwide/csm/documents/7972%20ULE%20Product%20Information%20Jan%202016.pdf
        kappa_conductivity=1.31,
        nu_poisson_ratio=0.17,
        beta_surface_absorption=1e-6,  # DUMMY
        intensity_reflectivity=1 - 100e-6 - 1e-6 - 10e-6,  # All - transmittance - absorption - scattering
        intensity_transmittance=100e-6,  # DUMMY - for mirrors
    ),
    "thermal_properties_fused_silica": MaterialProperties(
        refractive_index=1.455,  # https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson,
        alpha_expansion=0.48e-6,  # https://www.rp-photonics.com/fused_silica.html#:~:text=However%2C%20fused%20silica%20may%20exhibit,10%E2%88%926%20K%E2%88%921., https://www.swiftglass.com/blog/material-month-fused-silica/
        beta_surface_absorption=1e-6,  # DUMMY
        kappa_conductivity=1.38,  # https://www.swiftglass.com/blog/material-month-fused-silica/, https://www.heraeus-conamic.com/knowlegde-base/properties
        dn_dT=12e-6,  # https://iopscience.iop.org/article/10.1088/0022-3727/16/5/002/pdf
        nu_poisson_ratio=0.15,
        alpha_volume_absorption=1e-3,  # https://www.crystran.co.uk/optical-materials/silica-glass-sio2
        intensity_reflectivity=100e-6,  # DUMMY - for lenses
        intensity_transmittance=1 - 100e-6 - 1e-6,  # DUMMY - for lenses
    ),  # https://www.azom.com/properties.aspx?ArticleID=1387),
    "thermal_properties_yag": MaterialProperties(
        refractive_index=1.81,
        alpha_expansion=8e-6,  # https://www.crystran.co.uk/optical-materials/yttrium-aluminium-garnet-yag
        beta_surface_absorption=1e-6,  # DUMMY
        kappa_conductivity=11.2,  # https://www.scientificmaterials.com/downloads/Nd_YAG.pdf, This does not agree: https://pubs.aip.org/aip/jap/article/131/2/020902/2836262/Thermal-conductivity-and-management-in-laser-gain
        dn_dT=9e-6,  # https://pubmed.ncbi.nlm.nih.gov/18319922/
        nu_poisson_ratio=0.25,  #  https://www.crystran.co.uk/userfiles/files/yttrium-aluminium-garnet-yag-data-sheet.pdf, https://www.korth.de/en/materials/detail/YAG
    ),
    "thermal_properties_bk7": MaterialProperties(alpha_expansion=7.1e-6, kappa_conductivity=1.114),
    "c_mirror_radius_expansion": 1,  # DUMMY temp - should be 4 according to Lidan's simulation
    # But we take it to be 1 as the other values are currently 1.
    "c_lens_focal_length_expansion": 1,  # DUMMY
    "c_lens_volumetric_absorption": 1,  # DUMMY
}


def convert_material_to_mirror_or_lens(
    material_properties: MaterialProperties,
    convert_to_type: str,
    intensity_transmittance: Optional[float] = None,
    intensity_reflectivity: Optional[float] = None,
    scattering: Optional[float] = None,
) -> MaterialProperties:
    # The defaults are like so with the nvl because there are two defaults - for lenses and for mirrors.
    if convert_to_type.lower() == "lens":
        intensity_reflectivity = nvl(intensity_reflectivity, 100e-6)
        intensity_transmittance = 1 - material_properties.beta_surface_absorption - intensity_reflectivity
    elif convert_to_type.lower() == "mirror":
        scattering = nvl(scattering, 10e-6)
        intensity_transmittance = nvl(intensity_transmittance, 100e-6)
        intensity_reflectivity = 1 - scattering - material_properties.beta_surface_absorption - intensity_transmittance
    else:
        raise ValueError("convert_to_type argument must be either 'lens' or 'mirror'")

    material_properties.intensity_reflectivity = intensity_reflectivity
    material_properties.intensity_transmittance = intensity_transmittance

    return material_properties


@dataclass
class OpticalElementParams:
    surface_type: Union[str]
    x: float  # Of the center of the surface (the estimated point of intersection with the central line of the beam)
    y: float
    z: float
    theta: float  # the out-of-plane angle normal vector to the surface. when the plane is x,y, this is the theta angle.
    phi: float  # the in-plane angle normal vector to the surface. when the plane is x,y, this is the phi angle.
    r_1: float  # radius of curvature. np.inf for flat surfaces.
    r_2: float  # nan if the optical object has only one face, or if the two faces are fixed to the same radius of curvature.
    curvature_sign: int  # 1 if the surface is concave, -1 if it is convex  # ATTENTION: ONCE CONCAVE ELEMENTS WILL BE USED, THERE WILL HAVE TO BE TWO CURVATURE SIGNS
    T_c: float  # center thickness of the element
    n_inside_or_after: (
        float  # refractive index inside the optical object (for a ThickLens) or after it (for a refractive surface)
    )
    n_outside_or_before: (
        float  # refractive index outside the optical object (for a ThickLens) or before it (for a refractive surface)
    )
    material_properties: MaterialProperties

    # def __post_init__(self):
    #     assert self.material_properties.refractive_index == self.n_inside_or_after or self.material_properties.refractive_index == self.n_outside_or_before or np.isnan(self.material_properties.refractive_index), "The refractive index of the material properties is neither of the refractive indices of the optical element!"

    def __repr__(self):
        surface_type_string = f"'{self.surface_type}'"
        surface_type_string = surface_type_string.ljust(27)
        curvature_sign_string = "CurvatureSigns.concave" if self.curvature_sign == -1 else "CurvatureSigns.convex"
        return (
            f"OpticalElementParams("
            f"surface_type={surface_type_string}, "
            f"x={pretty_print_number(self.x)}, "
            f"y={pretty_print_number(self.y)}, "
            f"z={pretty_print_number(self.z)}, "
            f"theta={pretty_print_number(self.theta, represents_angle=True)}, "
            f"phi={pretty_print_number(self.phi, represents_angle=True)}, "
            f"r_1={pretty_print_number(self.r_1)}, "
            f"r_2={pretty_print_number(self.r_2)}, "
            f"curvature_sign={curvature_sign_string}, "
            f"T_c={pretty_print_number(self.T_c)}, "
            f"n_inside_or_after={pretty_print_number(self.n_inside_or_after)}, "
            f"n_outside_or_before={pretty_print_number(self.n_outside_or_before)}, "
            f"material_properties={self.material_properties})"
        )

    @property
    def to_array(self) -> np.ndarray:
        if isinstance(self.surface_type, str):
            surface_type = SURFACE_TYPES_DICT[self.surface_type]
        else:
            surface_type = self.surface_type  # This should not happen
        array = np.zeros(len(INDICES_DICT.keys())).astype(np.complex128)
        array[
            [
                INDICES_DICT["surface_type"],
                INDICES_DICT["x"],
                INDICES_DICT["y"],
                INDICES_DICT["z"],
                INDICES_DICT["theta"],
                INDICES_DICT["phi"],
                INDICES_DICT["r_1"],
                INDICES_DICT["r_2"],
                INDICES_DICT["curvature_sign"],
                INDICES_DICT["T_c"],
                INDICES_DICT["n_inside_or_after"],
                INDICES_DICT["n_outside_or_before"],
                INDICES_DICT["material_refractive_index"],
                INDICES_DICT["alpha_expansion"],
                INDICES_DICT["beta_surface_absorption"],
                INDICES_DICT["kappa_conductivity"],
                INDICES_DICT["dn_dT"],
                INDICES_DICT["nu_poisson_ratio"],
                INDICES_DICT["alpha_volume_absorption"],
                INDICES_DICT["intensity_reflectivity"],
                INDICES_DICT["intensity_transmittance"],
                INDICES_DICT["temperature"],
            ]
        ] = [
            surface_type,
            self.x,
            self.y,
            self.z,
            self.theta,
            self.phi,
            self.r_1,
            self.r_2,
            self.curvature_sign,
            self.T_c,
            self.n_inside_or_after,
            self.n_outside_or_before,
            *self.material_properties.to_array,
        ]
        array[INDICES_DICT["theta"]] = 1j * array[INDICES_DICT["theta"]] / np.pi
        array[INDICES_DICT["phi"]] = 1j * array[INDICES_DICT["phi"]] / np.pi
        return array

    @staticmethod
    def from_array(params: np.ndarray):
        params = np.real(params) + np.pi * np.imag(params)
        material_properties = MaterialProperties(
            refractive_index=nvl(
                params[INDICES_DICT["material_refractive_index"]],
                params[INDICES_DICT["n_inside_or_after"]],
            ),
            alpha_expansion=params[INDICES_DICT["alpha_expansion"]],
            beta_surface_absorption=params[INDICES_DICT["beta_surface_absorption"]],
            kappa_conductivity=params[INDICES_DICT["kappa_conductivity"]],
            dn_dT=params[INDICES_DICT["dn_dT"]],
            nu_poisson_ratio=params[INDICES_DICT["nu_poisson_ratio"]],
            alpha_volume_absorption=params[INDICES_DICT["alpha_volume_absorption"]],
            intensity_reflectivity=params[INDICES_DICT["intensity_reflectivity"]],
            intensity_transmittance=params[INDICES_DICT["intensity_transmittance"]],
            temperature=params[INDICES_DICT["temperature"]],
        )
        return OpticalElementParams(
            surface_type=SurfacesTypes.from_integer_representation(params[INDICES_DICT["surface_type"]]),
            x=params[INDICES_DICT["x"]],
            y=params[INDICES_DICT["y"]],
            z=params[INDICES_DICT["z"]],
            theta=params[INDICES_DICT["theta"]],
            phi=params[INDICES_DICT["phi"]],
            r_1=params[INDICES_DICT["r_1"]],
            r_2=params[INDICES_DICT["r_2"]],
            curvature_sign=params[INDICES_DICT["curvature_sign"]],
            T_c=params[INDICES_DICT["T_c"]],
            n_inside_or_after=params[INDICES_DICT["n_inside_or_after"]],
            n_outside_or_before=params[INDICES_DICT["n_outside_or_before"]],
            material_properties=material_properties,
        )


def params_to_perturbable_params_names(
    params_list: List[OpticalElementParams], remove_one_of_the_angles: bool = False
) -> List[str]:
    # Associates the cavity parameters with the number of parameters needed to describe the cavity.
    # If there is a lens, then the number of parameters is 7 (x, y, theta, phi, r, n_2):

    perturbable_params = [
        ParamsNames.x,
        ParamsNames.y,
        ParamsNames.theta,
        ParamsNames.phi,
        ParamsNames.r_1,
        ParamsNames.n_inside_or_after,
    ]

    surface_types = [params.surface_type for params in params_list]
    if not (
        SurfacesTypes.curved_refractive_surface in surface_types
        or SurfacesTypes.thick_lens in surface_types
        or SurfacesTypes.ideal_thick_lens in surface_types
    ):
        perturbable_params.remove(ParamsNames.n_inside_or_after)
    if remove_one_of_the_angles:
        perturbable_params.remove(ParamsNames.theta)
    return perturbable_params


# Throughout the code, all tensors can take any number of dimensions, but the last dimension is always the coordinate
# dimension. this allows a Ray to be either a single ray, a list of rays, or a list of lists of rays, etc.
# For example, a Ray could be a set of rays with a starting point for every combination of x, y, z. in this case, the
# ray.origin tensor will be of the size N_x | N_y | N_z | 3.

# The ray is always traced starting from the last surface of the cavity, such that the first mirror is the first mirror
# the ray hits. in the initial state of the cavity it means that the ray starts from cavity.physical_surfaces[-1].center and hits
# first the cavity.physical_surfaces[0] mirror. After the plane that is perpendicular to the central line and between the two
# physical_surfaces is calculated, then the ray starts at cavity.surfaces[-1].center (which is that plane) and hits first the
# cavity.surfaces[0] which is the first mirror.

# As a convention, the locations (parameterized usually by theta and phi) always appear before the angles (parameterized by
# theta and phi). also, theta and theta appear before phi and phi.
# If for example there is a parameter q both for theta axis and phi axis, then the first element of q will be the q of theta,
# and the second element of q will be the q of phi.


class LocalModeParameters:
    def __init__(
        self,
        z_minus_z_0: Optional[Union[np.ndarray, float]] = None,  # The actual distance should be multiplied by n
        z_R: Optional[Union[np.ndarray, float]] = None,
        q: Optional[Union[np.ndarray, float]] = None,
        lambda_0_laser: Optional[float] = None,  # the laser's wavelength in vacuum
        n: float = 1,  # refractive index
    ):
        if q is not None:
            if isinstance(q, float):
                q = np.array([q, q])
            self.q: np.ndarray = q
        elif z_minus_z_0 is not None and z_R is not None:
            if isinstance(z_R, float):
                z_R = np.array([z_R, z_R])
            if isinstance(q, float):
                z_minus_z_0 = np.array([z_minus_z_0, z_minus_z_0])
            self.q: np.ndarray = z_minus_z_0 + 1j * z_R
        else:
            raise ValueError("Either q or z_minus_z_0 and z_R must be provided")
        self.lambda_0_laser = lambda_0_laser
        self.n = n

    @property
    def z_minus_z_0(self):
        return self.q.real

    @property
    def z_R(self):
        return self.q.imag

    @property
    def w_0(self):
        return w_0_of_z_R(z_R=self.z_R, lambda_0_laser=self.lambda_0_laser, n=self.n)

    @property
    def lambda_laser(self):
        return self.lambda_0_laser / self.n

    def to_mode_parameters(self, location_of_local_mode_parameter: np.ndarray, k_vector: np.ndarray):
        center = location_of_local_mode_parameter - self.z_minus_z_0[:, np.newaxis] * k_vector
        z_hat = np.array([0, 0, 1])
        if np.linalg.norm(k_vector - z_hat) < 1e-10:  # if the k_vector is almost parallel to z_hat, better take another
            # vector as z_hat to avoid numerical instability
            z_hat = np.array([0, 1, 0])
        pseudo_y = normalize_vector(np.cross(z_hat, k_vector))
        pseudo_z = normalize_vector(np.cross(k_vector, pseudo_y))
        principle_axes = np.stack([pseudo_z, pseudo_y], axis=0)

        return ModeParameters(
            center=center,
            k_vector=k_vector,
            w_0=self.w_0,
            principle_axes=principle_axes,
            lambda_0_laser=self.lambda_0_laser,
        )

    @property
    def spot_size(self):
        if np.any(self.z_R == 0):
            w_z = np.array([np.nan, np.nan])
        else:
            w_z = spot_size(z=self.z_minus_z_0, z_R=self.z_R, lambda_0_laser=self.lambda_0_laser, n=self.n)  # spot_size
        return w_z


@dataclass
class ModeParameters:
    # I have once spent a few hours unifying LocalModeParameters and ModeParameters into one class and at the end
    # saw it only makes the code more cumbersome and less readable, so I rolled back.
    center: (
        np.ndarray
    )  # First dimension is theta or phi (the two transversal axes of the mode), second dimension is x, y, z
    k_vector: np.ndarray
    w_0: np.ndarray
    principle_axes: np.ndarray  # First dimension is theta or phi, second dimension is x, y, z
    lambda_0_laser: Optional[float]
    n: float = 1  # refractive index

    @property
    def ray(self):
        return Ray(self.center, self.k_vector)

    @property
    def z_R(self):  # The Rayleigh range in vacuum
        if self.lambda_0_laser is None:
            return None
        else:
            return np.pi * self.w_0**2 / self.lambda_0_laser

    @property
    def lambda_laser(self):
        return self.lambda_0_laser / self.n

    @property
    def NA(self):
        if self.lambda_0_laser is None:
            return None
        else:
            if self.z_R[0] == 0 or self.z_R[1] == 0:
                return np.array([np.nan, np.nan])
            else:
                return np.sqrt(self.lambda_0_laser / (np.pi * self.z_R))

    def local_mode_parameters(self, z_minus_z_0):
        return LocalModeParameters(z_minus_z_0=z_minus_z_0, z_R=self.z_R, lambda_0_laser=self.lambda_0_laser, n=self.n)


def decompose_ABCD_matrix(
    ABCD: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if ABCD.shape == (4, 4):
        A, B, C, D = (
            ABCD[(0, 2), (0, 2)],
            ABCD[(0, 2), (1, 3)],
            ABCD[(1, 3), (0, 2)],
            ABCD[(1, 3), (1, 3)],
        )
    else:
        A, B, C, D = ABCD[0, 0], ABCD[0, 1], ABCD[1, 0], ABCD[1, 1]
    return A, B, C, D


def propagate_local_mode_parameter_through_ABCD(
    local_mode_parameters: LocalModeParameters, ABCD: np.ndarray, n_1: float = 1, n_2: float = 1
) -> LocalModeParameters:
    A, B, C, D = decompose_ABCD_matrix(ABCD)
    q_new = n_2 * (A * local_mode_parameters.q / n_1 + B) / (C * local_mode_parameters.q / n_1 + D)
    # q_new = (A * local_mode_parameters.q + B) / (C * local_mode_parameters.q + D)
    return LocalModeParameters(q=q_new, lambda_0_laser=local_mode_parameters.lambda_0_laser, n=n_2)


def local_mode_parameters_of_round_trip_ABCD(
    round_trip_ABCD: np.ndarray,
    n: float,  # refractive_index at the begining of the roundtrip
    lambda_0_laser: Optional[float] = None,
) -> LocalModeParameters:
    A, B, C, D = decompose_ABCD_matrix(round_trip_ABCD)
    q_z = (A - D + np.sqrt(A**2 + 2 * C * B + D**2 - 2 + 0j)) / (2 * C)
    q_z = np.real(q_z) + 1j * np.abs(np.imag(q_z))  # For the beam amplitude to decay with transverse radius and not
    # to grow, the term -ik * i*im(1/q_z) has to be negative. since in the simulation k is always positive, we take the
    # imaginary part of q_z to be positive (and the the imaginary part of it's inverse is negative).

    return LocalModeParameters(
        q=q_z, lambda_0_laser=lambda_0_laser, n=n
    )  # First dimension is theta or phi,second dimension is z_minus_z_0 or
    # z_R.


class Ray:
    def __init__(
        self,
        origin: np.ndarray,  # [m_rays..., 3]
        k_vector: np.ndarray,  # [m_rays..., 3]
        length: Optional[Union[np.ndarray, float]] = None,  # [m_rays..., 3]
    ):
        if k_vector.ndim == 1 and origin.shape[0] > 1:
            k_vector = np.tile(k_vector, (*origin.shape[:-1], 1))
        elif origin.ndim == 1 and k_vector.shape[0] > 1:
            origin = np.tile(origin, (*k_vector.shape[:-1], 1))

        self.origin = origin  # m_rays | 3
        self.k_vector = normalize_vector(k_vector)  # m_rays | 3
        if length is not None and isinstance(length, float) and origin.ndim > 1:  # If there is one length for many rays
            length = np.ones(origin.shape[0]) * length
        self.length = length  # m_rays or None

    def __getitem__(self, key):
        subscripted_ray = Ray(
            self.origin[key], self.k_vector[key], self.length[key] if self.length is not None else None
        )
        return subscripted_ray

    def parameterization(self, t: Union[np.ndarray, float]) -> np.ndarray:
        # Currently this function allows only one theta per ray. if needed it can be extended to allow multiple theta per ray.
        # theta needs to be either a float or a numpy array with dimensions m_rays
        return self.origin + t[..., np.newaxis] * self.k_vector

    def plot(self, ax: Optional[plt.Axes] = None, dim=2, plane: str = "xy", **kwargs):
        if ax is None:
            fig = plt.figure()
            if dim == 3:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.add_subplot(111)
        if self.length is None:
            length = np.ones_like(self.origin[..., 0])
        else:
            length = self.length
        ray_origin_reshaped = self.origin.reshape(-1, 3)
        ray_k_vector_reshaped = self.k_vector.reshape(-1, 3)
        lengths_reshaped = length.reshape(-1)
        if dim == 3:
            [
                ax.plot(
                    [
                        ray_origin_reshaped[i, 0],
                        ray_origin_reshaped[i, 0] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 0],
                    ],
                    [
                        ray_origin_reshaped[i, 1],
                        ray_origin_reshaped[i, 1] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 1],
                    ],
                    [
                        ray_origin_reshaped[i, 2],
                        ray_origin_reshaped[i, 2] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 2],
                    ],
                    **kwargs,
                )
                for i in range(ray_origin_reshaped.shape[0])
            ]
        else:
            x_index, y_index = plane_name_to_xy_indices(plane)
            [
                ax.plot(
                    [
                        ray_origin_reshaped[i, x_index],
                        ray_origin_reshaped[i, x_index] + lengths_reshaped[i] * ray_k_vector_reshaped[i, x_index],
                    ],
                    [
                        ray_origin_reshaped[i, y_index],
                        ray_origin_reshaped[i, y_index] + lengths_reshaped[i] * ray_k_vector_reshaped[i, y_index],
                    ],
                    **kwargs,
                )
                for i in range(ray_origin_reshaped.shape[0])
            ]

        return ax


class Surface:
    def __init__(
        self,
        outwards_normal: np.ndarray,
        radius: float,
        name: Optional[str] = None,
        material_properties: MaterialProperties = None,
        **kwargs,
    ):
        self.outwards_normal = normalize_vector(outwards_normal)
        self.name = name
        self.radius = radius
        self.material_properties = material_properties

    @property
    def center(self):
        raise NotImplementedError

    @property
    def inwards_normal(self):
        return -self.outwards_normal

    def find_intersection_with_ray(self, ray: Ray, paraxial: bool = False) -> np.ndarray:
        if paraxial:
            return self.find_intersection_with_ray_paraxial(ray)
        else:
            return self.find_intersection_with_ray_exact(ray)

    def find_intersection_with_ray_paraxial(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError

    def find_intersection_with_ray_exact(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        # Take parameters and return points on the surface
        raise NotImplementedError

    def get_parameterization(self, points: np.ndarray):
        # takes a point on the surface and returns the parameters
        raise NotImplementedError

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        name: Optional[str] = None,
        dim: int = 2,
        length=0.6,
        plane: str = "xy",
    ):
        if ax is None:
            fig = plt.figure()
            if dim == 3:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.add_subplot(111)
        if dim == 3:
            s = np.linspace(-length / 2, length / 2, 100)
            t = np.linspace(-length / 2, length / 2, 100)
        else:
            if plane in ["xy", "yx"]:
                t = 0
                s = np.linspace(-length / 2, length / 2, 100)
            elif plane in ["xz", "zx"]:
                s = 0
                t = np.linspace(-length / 2, length / 2, 100)
            elif plane in ["yz", "zy"]:
                s = 0
                t = np.linspace(-length / 2, length / 2, 100)
            else:
                raise ValueError("plane must be one of 'xy', 'xz', 'yz'")

        T, S = np.meshgrid(t, s)
        points = self.parameterization(T, S)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        if isinstance(self, CurvedRefractiveSurface):
            color = "grey"
        elif isinstance(self, PhysicalSurface):
            color = "b"
        else:
            color = "black"

        if dim == 3:
            ax.plot_surface(x, y, z, color=color, alpha=0.25)
        else:
            if plane in ["xy", "yx"]:
                x_dummy = points[:, 0, 0]
                y_dummy = points[:, 0, 1]
            elif plane in ["xz", "zx"]:
                x_dummy = points[0, :, 0]
                y_dummy = points[0, :, 2]
            elif plane in ["yz", "zy"]:
                x_dummy = points[0, :, 1]
                y_dummy = points[0, :, 2]
            else:
                raise ValueError("plane must be one of 'xy', 'xz', 'yz'")
            ax.plot(x_dummy, y_dummy, color=color)
        if name is not None:
            name_position = self.parameterization(0.4, 0)
            if dim == 3:
                ax.text(name_position[0], name_position[1], name_position[2], s=name)
            else:
                if (
                    ax.get_xlim()[0] < name_position[0] < ax.get_xlim()[1]
                    and ax.get_ylim()[0] < name_position[1] < ax.get_ylim()[1]
                ):
                    ax.text(name_position[0], name_position[1], s=name)
        if plane in ["xy", "yx"]:
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
        elif plane in ["xz", "zx"]:
            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m]")
        elif plane in ["yz", "zy"]:
            ax.set_xlabel("y [m]")
            ax.set_ylabel("z [m]")
        if dim == 3:
            ax.set_zlabel("z [m]")

        # center_plus_normal = self.center + self.inwards_normal * length
        # if dim == 3:
        #     ax.plot([self.center[0], center_plus_normal[0]],
        #             [self.center[1], center_plus_normal[1]],
        #             [self.center[2], center_plus_normal[2]], 'g-')
        # else:
        #     ax.plot([self.center[0], center_plus_normal[0]],
        #             [self.center[1], center_plus_normal[1]], 'g-')
        return ax

    def generate_ray_from_parameters(self, t: float, p: float, theta: float, phi: float) -> Ray:
        k_vector = unit_vector_of_angles(theta, phi)
        origin = self.parameterization(t, p)
        return Ray(origin=origin, k_vector=k_vector)

    def spanning_vectors(self):
        # Should change it to treat the case where inwards_normal is parallel to z_hat
        pseudo_y = normalize_vector(np.cross(np.array([0, 0, 1]), self.inwards_normal))
        pseudo_z = normalize_vector(np.cross(self.inwards_normal, pseudo_y))
        return pseudo_z, pseudo_y

    @staticmethod
    def from_params(params: OpticalElementParams, name: Optional[str] = None):
        p = params
        center = np.array([p.x, p.y, p.z])
        outwards_normal = unit_vector_of_angles(p.theta, p.phi)
        if p.surface_type == SurfacesTypes.curved_mirror:  # Mirror
            surface = CurvedMirror(
                radius=p.r_1,
                outwards_normal=outwards_normal,
                center=center,
                curvature_sign=p.curvature_sign,
                name=name,
                thermal_properties=p.material_properties,
            )
        elif p.surface_type == SurfacesTypes.thick_lens:  # ThickLens
            surface = generate_lens_from_params(p, name=name)
        elif p.surface_type == SurfacesTypes.ideal_thick_lens:  # IdealThickLens
            surface = generate_thick_ideal_lens_from_params(p, name=name)
        elif p.surface_type == SurfacesTypes.curved_refractive_surface:  # Refractive surface (one side of a lens)
            surface = CurvedRefractiveSurface(
                radius=p.r_1,
                outwards_normal=outwards_normal,
                center=center,
                n_1=p.n_outside_or_before,
                n_2=p.n_inside_or_after,
                curvature_sign=p.curvature_sign,
                name=name,
                thermal_properties=p.material_properties,
                thickness=p.T_c,
            )
        elif p.surface_type == SurfacesTypes.ideal_lens:  # Ideal lens
            surface = IdealLens(
                outwards_normal=outwards_normal,
                center=center,
                focal_length=p.r_1,
                name=name,
                thermal_properties=p.material_properties,
            )
        elif p.surface_type == SurfacesTypes.flat_mirror:  # Flat mirror
            surface = FlatMirror(
                outwards_normal=outwards_normal,
                center=center,
                name=name,
                thermal_properties=p.material_properties,
            )
        else:
            raise ValueError(f"Unknown surface type {p.surface_type}")
        return surface

    @property
    def to_params(self) -> OpticalElementParams:
        x, y, z = self.center
        if isinstance(self, IdealLens):
            r_1 = self.focal_length
            r_2 = np.nan
        elif isinstance(self, CurvedSurface):
            r_1 = self.radius
            r_2 = np.nan
        else:
            r_1 = 0
            r_2 = 0
        theta, phi = angles_of_unit_vector(self.outwards_normal)
        n_1 = 1
        n_2 = 1
        if isinstance(self, CurvedMirror):
            surface_type = SurfacesTypes.curved_mirror
            curvature_sign = self.curvature_sign
        elif isinstance(self, CurvedRefractiveSurface):
            surface_type = SurfacesTypes.curved_refractive_surface
            n_1 = self.n_1
            n_2 = self.n_2
            curvature_sign = self.curvature_sign
        elif isinstance(self, IdealLens):
            surface_type = SurfacesTypes.ideal_lens
            curvature_sign = 0
        elif isinstance(self, FlatMirror):
            surface_type = SurfacesTypes.flat_mirror
            curvature_sign = 0
        else:
            raise ValueError(f"Unknown surface type {type(self)}")
        if self.material_properties is None:
            self.material_properties = MaterialProperties()

        params = OpticalElementParams(
            surface_type=surface_type,
            x=x,
            y=y,
            z=z,
            theta=theta,
            phi=phi,
            r_1=r_1,
            r_2=r_2,
            curvature_sign=curvature_sign,
            T_c=np.nan,
            n_inside_or_after=n_2,
            n_outside_or_before=n_1,
            material_properties=self.material_properties,
        )
        return params


class PhysicalSurface(Surface):
    def __init__(
        self,
        outwards_normal: np.ndarray,
        radius: float,
        name: Optional[str] = None,
        material_properties: Optional[MaterialProperties] = None,
        **kwargs,
    ):

        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            radius=radius,
            material_properties=material_properties,
            **kwargs,
        )

    @property
    def center(self):
        raise NotImplementedError

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        raise NotImplementedError

    def get_parameterization(self, points: np.ndarray):
        raise NotImplementedError

    def reflect_ray(self, ray: Ray, paraxial: bool = False) -> Ray:
        intersection_point = self.find_intersection_with_ray(ray, paraxial=paraxial)
        reflected_direction_vector = self.reflect_direction(ray, paraxial=paraxial)
        return Ray(intersection_point, reflected_direction_vector)

    def reflect_direction(self, ray: Ray, paraxial: bool = False) -> np.ndarray:
        if paraxial:
            return self.reflect_direction_paraxial(ray)
        else:
            return self.reflect_direction_exact(ray)

    def reflect_direction_paraxial(self, ray: Ray) -> np.ndarray:
        if ray.k_vector.reshape(-1)[0:3] @ self.outwards_normal > 0:
            forwards_normal = self.outwards_normal
        else:
            forwards_normal = -self.outwards_normal

        flat_surface = FlatSurface(
            outwards_normal=forwards_normal,
            center=self.center,
        )
        intersection_point = flat_surface.find_intersection_with_ray(ray, paraxial=True)
        pseudo_z, pseudo_y = flat_surface.spanning_vectors()
        t, p = flat_surface.get_parameterization(
            intersection_point
        )  # Those are the coordinates of pseudo_z and pseudo_y
        t_projection, p_projection = ray.k_vector @ pseudo_z, ray.k_vector @ pseudo_y
        theta, phi = np.pi / 2 - np.arccos(t_projection), np.pi / 2 - np.arccos(p_projection)
        input_vector = np.array([t, theta, p, phi])
        if len(input_vector.shape) > 1:
            input_vector = np.swapaxes(input_vector, 0, 1)
        output_vector = self.ABCD_matrix(cos_theta_incoming=1) @ input_vector  # For the sake of ray tracing, we
        # reflect the ray with respect to the optical element's optical axis, and not with respect to the central line
        # that was even not calculate yet. therefore, the cos_theta_incoming used here is the trivial one.
        if len(input_vector.shape) > 1:
            output_vector = np.swapaxes(output_vector, 0, 1)
        t_projection_out, p_projection_out = cos_without_trailing_epsilon(
            np.pi / 2 - output_vector[1, ...]
        ), cos_without_trailing_epsilon(np.pi / 2 - output_vector[3, ...])
        # Those are the components of the output direction vector in the pseudo_z and pseudo_y and
        # surface_normal directions:
        component_t = np.multiply.outer(t_projection_out, pseudo_z)
        component_p = np.multiply.outer(p_projection_out, pseudo_y)
        component_n = np.multiply.outer((1 - t_projection_out**2 - p_projection_out**2) ** 0.5, forwards_normal)
        output_direction_vector = component_t + component_p + component_n
        return output_direction_vector

    def reflect_direction_exact(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError

    def ABCD_matrix(self, cos_theta_incoming: Optional[float] = None) -> np.ndarray:
        raise NotImplementedError

    def thermal_transformation(self, P_laser_power: float, w_spot_size: float, **kwargs):
        raise NotImplementedError


class FlatSurface(Surface):
    def __init__(
        self,
        outwards_normal: np.ndarray,
        distance_from_origin: Optional[float] = None,
        center: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(outwards_normal=outwards_normal, name=name, radius=np.inf, **kwargs)
        if distance_from_origin is None and center is None:
            raise ValueError("Either distance_from_origin or center must be specified")
        if distance_from_origin is not None and center is not None:
            raise ValueError("Only one of distance_from_origin or center must be specified")
        if distance_from_origin is not None:
            self.distance_from_origin = distance_from_origin
            self.center_of_mirror_private = self.outwards_normal * distance_from_origin
        if center is not None:
            self.center_of_mirror_private = center
            self.distance_from_origin = center @ self.outwards_normal

    def find_intersection_with_ray_exact(self, ray: Ray) -> np.ndarray:
        A_vec = self.outwards_normal * self.distance_from_origin
        BA_vec = A_vec - ray.origin
        BC = BA_vec @ self.outwards_normal
        cos_theta = ray.k_vector @ self.outwards_normal
        t = BC / cos_theta
        intersection_point = ray.parameterization(t)
        ray.length = np.linalg.norm(intersection_point - ray.origin, axis=-1)
        return intersection_point

    def find_intersection_with_ray_paraxial(self, ray: Ray) -> np.ndarray:
        # Notes are available here: https://mynotebook.labarchives.com/MTM3NjE3My41fDEwNTg1OTUvMTA1ODU5NS9Ob3RlYm9vay8zMjQzMzA0MzY1fDM0OTMzNjMuNQ==/page/11290221-36
        cos_theta = self.outwards_normal @ ray.k_vector
        if cos_theta > 0:
            forwards_normal = self.outwards_normal
        else:
            forwards_normal = self.inwards_normal
            cos_theta = np.abs(cos_theta)
        sin_abs_theta = np.linalg.norm(np.cross(ray.k_vector, forwards_normal))
        theta = np.arcsin(sin_abs_theta)  # You might as - but wait, can't we use the arccos of the cos_theta already
        # calculated? the answer is no, because d/dx(cos) is 0 around 0 and d/dx(arccos) is infinite around 0, Which
        # leads to numerical instability when dealing with small angles.

        closest_point_in_plane_to_global_origin = self.distance_from_origin * self.outwards_normal  # v in notes

        displacement_in_plane = ray.origin - (forwards_normal @ ray.origin) * forwards_normal

        ray_origin_projected_onto_plane = (
            closest_point_in_plane_to_global_origin + displacement_in_plane
        )  # p_r in notes

        distance_between_rays_origin_and_plane = np.abs(
            self.distance_from_origin - (self.outwards_normal @ ray.origin)
        )  # h in notes

        vector_in_plane_in_k_n_plane = ray.k_vector - cos_theta * forwards_normal  # u in notes
        if np.linalg.norm(vector_in_plane_in_k_n_plane) < 1e-20:
            return ray_origin_projected_onto_plane
        else:
            vector_in_plane_in_k_n_plane = normalize_vector(vector_in_plane_in_k_n_plane)
            intersection_point = (
                ray_origin_projected_onto_plane
                + theta * distance_between_rays_origin_and_plane * vector_in_plane_in_k_n_plane
            )
        return intersection_point

    @property
    def center(self):
        # The reason for this property is that in other PhysicalSurface classes it is a property.
        return self.center_of_mirror_private

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]):
        pseudo_z, pseudo_y = self.spanning_vectors()
        if isinstance(t, (float, int)):
            t = np.array(t)
        if isinstance(p, (float, int)):
            p = np.array(p)
        points = self.center + t[..., np.newaxis] * pseudo_z + p[..., np.newaxis] * pseudo_y
        return points

    def get_parameterization(self, points: np.ndarray):
        pseudo_z, pseudo_y = self.spanning_vectors()
        t = (points - self.center) @ pseudo_z
        p = (points - self.center) @ pseudo_y
        return t, p


class FlatMirror(FlatSurface, PhysicalSurface):

    def __init__(
        self,
        outwards_normal: np.ndarray,
        distance_from_origin: Optional[float] = None,
        center: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        thermal_properties: Optional[MaterialProperties] = None,
    ):
        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            material_properties=thermal_properties,
            distance_from_origin=distance_from_origin,
            center=center,
            radius=np.inf,
        )

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        name: Optional[str] = None,
        dim: int = 3,
        length=0.6,
        plane: str = "xy",
    ):
        return super().plot(ax, name, dim, length, plane)

    def get_parameterization(self, points: np.ndarray):
        return super().get_parameterization(points)

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        return super().parameterization(t, p)

    @property
    def center(self):
        return super().center

    def reflect_direction_exact(self, ray: Ray) -> np.ndarray:
        dot_product = ray.k_vector @ self.outwards_normal  # m_rays
        k_projection_on_normal = dot_product[..., np.newaxis] * self.outwards_normal
        reflected_direction_test = ray.k_vector - 2 * k_projection_on_normal
        return reflected_direction_test

    def ABCD_matrix(self, cos_theta_incoming: float = None) -> np.ndarray:
        # Assumes the ray is in the x-y plane, and the mirror is in the z-x plane
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

    @property
    def radius(self):
        return np.inf

    def thermal_transformation(self, P_laser_power: float, w_spot_size: float):
        raise NotImplementedError


class FlatRefractiveSurface(FlatSurface, PhysicalSurface):

    def __init__(
        self,
        outwards_normal: np.ndarray,
        distance_from_origin: Optional[float] = None,
        center: Optional[np.ndarray] = None,
        n_1: float = 1,
        n_2: float = 1,
        name: Optional[str] = None,
        thermal_properties: Optional[MaterialProperties] = None,
    ):
        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            material_properties=thermal_properties,
            distance_from_origin=distance_from_origin,
            center=center,
            radius=np.inf,
        )
        self.n_1 = n_1
        self.n_2 = n_2

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        name: Optional[str] = None,
        dim: int = 3,
        length=0.6,
        plane: str = "xy",
    ):
        return super().plot(ax, name, dim, length, plane)

    def reflect_direction_exact(self, ray: Ray) -> np.ndarray:
        # Assumes self.outwards_normal is pointing towards the medium with n_2.
        # This uses the same derivation as in https://mynotebook.labarchives.com/MTM3NjE3My41fDEwNTg1OTUvMTA1ODU5NS9Ob3RlYm9vay8zMjQzMzA0MzY1fDM0OTMzNjMuNQ==/page/11290221-33
        # except that here n_forwards and n_backwards are constants.
        cos_theta_incoming = np.clip(np.sum(ray.k_vector * self.outwards_normal, axis=-1), a_min=-1, a_max=1)  # m_rays
        n_orthogonal = (  # This is the vector that is orthogonal to the normal to the surface and lives in the plane spanned by the ray and the normal to the surface (Grahm-Schmidt process).
            ray.k_vector - cos_theta_incoming[..., np.newaxis] * self.outwards_normal
        )  # m_rays | 3
        n_orthogonal_norm = np.linalg.norm(n_orthogonal, axis=-1)  # m_rays
        if isinstance(n_orthogonal_norm, float) and n_orthogonal_norm < 1e-15:
            reflected_direction_vector = self.outwards_normal
        else:
            practically_normal_incidences = n_orthogonal_norm < 1e-15
            n_orthogonal[practically_normal_incidences] = (
                np.nan
            )  # This is done so that the normalization does not throw an error. those values will later be filled with
            # the trivial solution, so no nans in the output.
            n_orthogonal = normalize_vector(n_orthogonal)
            sin_theta_incoming = np.sqrt(1 - cos_theta_incoming**2)
            sin_theta_outgoing = (self.n_1 / self.n_2) * sin_theta_incoming  # m_rays  # Snell's law
            cos_theta_outgoing = stable_sqrt(1 - sin_theta_outgoing**2)  # m_rays
            reflected_direction_vector = (
                self.outwards_normal
                * cos_theta_outgoing[..., np.newaxis]  # outward_normal * cos(theta_o) + n_orthogonal * sin(theta_o)
                + n_orthogonal * sin_theta_outgoing[..., np.newaxis]
            )  # m_rays | 3

            reflected_direction_vector[practically_normal_incidences] = self.outwards_normal[
                practically_normal_incidences
            ]  # For the nans we initiated before, we just want the normal to the surface to be the new direction of the ray
        return reflected_direction_vector

    def ABCD_matrix(self, cos_theta_incoming: float = None) -> np.ndarray:
        # Note ! this code assumes the ray is in the x-y plane! Until it is fixed, the only perturbations in x,y,phi should be calculated!
        sin_theta_incoming = np.sqrt(1 - cos_theta_incoming**2)
        sin_theta_outgoing = (self.n_1 / self.n_2) * sin_theta_incoming
        cos_theta_outgoing = stable_sqrt(1 - sin_theta_outgoing**2)
        return np.array(
            [
                [1, 0, 0, 0],
                [0, self.n_1 / self.n_2, 0],
                [0, 0, cos_theta_outgoing / cos_theta_incoming, 0],
                [0, 0, 0, (self.n_2 * cos_theta_incoming) / (self.n_1 * cos_theta_outgoing)],
            ]
        )


class IdealLens(FlatSurface, PhysicalSurface):
    def __init__(
        self,
        outwards_normal: np.ndarray,
        distance_from_origin: Optional[float] = None,
        center: Optional[np.ndarray] = None,
        focal_length: Optional[float] = None,
        name: Optional[str] = None,
        thermal_properties: Optional[MaterialProperties] = None,
    ):
        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            material_properties=thermal_properties,
            distance_from_origin=distance_from_origin,
            center=center,
        )
        self.focal_length = focal_length

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        name: Optional[str] = None,
        dim: int = 3,
        length=0.6,
        plane: str = "xy",
    ):
        return super().plot(ax, name, dim, length, plane)

    def get_parameterization(self, points: np.ndarray):
        return super().get_parameterization(points)

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        return super().parameterization(t, p)

    @property
    def center(self):
        return super().center

    @property
    def to_params(self) -> OpticalElementParams:
        raise NotImplementedError

    def reflect_direction_exact(self, ray: Ray) -> np.ndarray:
        intersection_point = self.find_intersection_with_ray(ray)
        pseudo_z, pseudo_y = self.spanning_vectors()
        t, p = self.get_parameterization(intersection_point)  # Those are the coordinates of pseudo_z and pseudo_y
        t_projection, p_projection = ray.k_vector @ pseudo_z, ray.k_vector @ pseudo_y
        theta, phi = np.pi / 2 - np.arccos(t_projection), np.pi / 2 - np.arccos(p_projection)
        input_vector = np.array([t, theta, p, phi])
        if len(input_vector.shape) > 1:
            input_vector = np.swapaxes(input_vector, 0, 1)
        output_vector = self.ABCD_matrix(cos_theta_incoming=1) @ input_vector
        if len(input_vector.shape) > 1:
            output_vector = np.swapaxes(output_vector, 0, 1)
        t_projection_out, p_projection_out = np.cos(np.pi / 2 - output_vector[1, ...]), np.cos(
            np.pi / 2 - output_vector[3, ...]
        )
        # ABCD_MATRIX METHOD
        # Here I assume all rays come from the same direction to the lens
        if ray.k_vector.reshape(-1)[0:3] @ self.outwards_normal > 0:
            forwards_normal = self.outwards_normal
        else:
            forwards_normal = -self.outwards_normal
        component_t = np.multiply.outer(t_projection_out, pseudo_z)
        component_p = np.multiply.outer(p_projection_out, pseudo_y)
        component_n = np.multiply.outer((1 - t_projection_out**2 - p_projection_out**2) ** 0.5, forwards_normal)
        output_direction_vector = component_t + component_p + component_n

        return output_direction_vector

    def ABCD_matrix(self, cos_theta_incoming: float = None) -> np.ndarray:
        # THIS CURRENTLY DOES NOT HOLD FOR THE CASE WHERE THE RAY IS NOT PERPENDICULAR TO THE LENS!
        ABCD = np.array(
            [
                [1, 0, 0, 0],
                [-1 / self.focal_length, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1 / self.focal_length, 1],
            ]
        )
        return ABCD

    def thermal_transformation(self, P_laser_power: float, w_spot_size: float):
        raise NotImplementedError


class CurvedSurface(Surface):
    def __init__(
        self,
        radius: float,
        outwards_normal: np.ndarray,  # Pointing from the origin of the sphere to the mirror's center.
        center: Optional[np.ndarray] = None,  # Not the center of the sphere but the center of
        # the plate.
        origin: Optional[np.ndarray] = None,  # The center of the sphere.
        curvature_sign: int = 1,
        # 1 for concave (where the ray is hitting the sphere from the inside) and -1 for convex
        # (where the ray is hitting the sphere from the outside). this is used to find the correct intersection
        # point of a ray with the surface
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(outwards_normal=outwards_normal, name=name, radius=radius, **kwargs)
        self.curvature_sign = curvature_sign
        if origin is None and center is None:
            raise ValueError("Either origin or center must be provided.")
        elif origin is not None and center is not None:
            raise ValueError("Only one of origin and center must be provided.")
        elif origin is None:
            self.origin = center + radius * self.inwards_normal
        else:
            self.origin = origin

    def find_intersection_with_ray_exact(self, ray: Ray) -> np.ndarray:
        # The following lines is the result of the next line of mathematica to find the intersection:
        # Solve[(x0 + kx * theta - xc) ^ 2 + (y0 + ky * theta - yc) ^ 2 + (z0 + kz * theta - zc) ^ 2 == R ^ 2, theta]
        l = (
            -ray.k_vector[..., 0] * ray.origin[..., 0]
            + ray.k_vector[..., 0] * self.origin[0]
            - ray.k_vector[..., 1] * ray.origin[..., 1]
            + ray.k_vector[..., 1] * self.origin[1]
            - ray.k_vector[..., 2] * ray.origin[..., 2]
            + ray.k_vector[..., 2] * self.origin[2]
            + self.curvature_sign
            * stable_sqrt(  # The stable_sqrt is to avoid numerical instability when the argument is negative.
                # it returns nans on negative values instead of throwing an error.
                -4
                * (ray.k_vector[..., 0] ** 2 + ray.k_vector[..., 1] ** 2 + ray.k_vector[..., 2] ** 2)
                * (
                    -self.radius**2
                    + (ray.origin[..., 0] - self.origin[0]) ** 2
                    + (ray.origin[..., 1] - self.origin[1]) ** 2
                    + (ray.origin[..., 2] - self.origin[2]) ** 2
                )
                + 4
                * (
                    ray.k_vector[..., 0] * (ray.origin[..., 0] - self.origin[0])
                    + ray.k_vector[..., 1] * (ray.origin[..., 1] - self.origin[1])
                    + ray.k_vector[..., 2] * (ray.origin[..., 2] - self.origin[2])
                )
                ** 2
            )
            / 2
        ) / (ray.k_vector[..., 0] ** 2 + ray.k_vector[..., 1] ** 2 + ray.k_vector[..., 2] ** 2)
        intersection_point = ray.parameterization(l)
        ray.length = l
        return intersection_point

    def find_intersection_with_ray_paraxial(self, ray: Ray) -> np.ndarray:
        flat_surface = FlatSurface(center=self.center, outwards_normal=self.outwards_normal)
        intersection_point = flat_surface.find_intersection_with_ray_paraxial(ray)
        ray.length = np.linalg.norm(intersection_point - ray.origin, axis=-1)
        return intersection_point

    def parameterization(
        self,
        t: Union[np.ndarray, float],  # the length of arc to travel on the sphere from the center
        # of the mirror to the point of interest, in the direction "pseudo_z". pseudo_z is
        # described in the get_spanning_vectors method. it is analogous to theta / R in the
        # classical parameterization.
        p: Union[np.ndarray, float],  # The same as theta but in the direction of pseudo_y. It is analogous
        # to phi / R in the classical parameterization.
    ) -> np.ndarray:
        # This parameterization treats the sphere as if as the center of the mirror was on the x-axis.
        # The conceptual difference between this parameterization and the classical one of [sin(theta)cos(phi),
        # sin(theta)sin(phi), cos(theta)]] is that here there is barely any Jacobian determinant.
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        # Notice how the order of rotations matters. First we rotate around the z axis, then around the y-axis.
        # Doing it the other way around would give parameterization that is not aligned with the conventional theta, phi
        # parameterization. This is important for the get_parameterization method.
        rotation_matrix = rotation_matrix_around_n(pseudo_y, -t / self.radius) @ rotation_matrix_around_n(
            pseudo_z, p / self.radius
        )  # The minus sign is because of the
        # orientation of the pseudo_y axis.

        points = self.origin + self.radius * rotation_matrix @ self.outwards_normal
        return points

    def get_parameterization(self, points: np.ndarray):
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        normalized_points = (points - self.origin) / self.radius
        p = np.arctan2(normalized_points @ pseudo_y, normalized_points @ self.outwards_normal) * self.radius
        # Notice that theta is like theta but instead of ranging in [0, pi] it ranges in [-pi/2, pi/2].
        t = np.arcsin(np.clip(normalized_points @ pseudo_z, -1, 1)) * self.radius
        return t, p

    @property
    def center(self):
        return self.origin + self.radius * self.outwards_normal

    def get_spanning_vectors(self):
        # For the case of the sphere with normal on the x-axis, those will be the y and z axis.
        # For the case of the sphere with normal on the y-axis, those will be the x and z axis.
        pseudo_y = np.cross(np.array([0, 0, 1]), self.inwards_normal)
        pseudo_z = np.cross(self.inwards_normal, pseudo_y)  # Should be approximately equal to \hat{z}, and exactly
        # equal if the outwards_normal is in the x-y plane.
        return pseudo_y, pseudo_z

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        name: Optional[str] = None,
        dim: int = 2,
        length=None,
        plane: str = "xy",
    ):
        if length is None:
            length = 0.6 * self.radius
        super().plot(ax, name, dim, length=length, plane=plane)


class CurvedMirror(CurvedSurface, PhysicalSurface):
    def __init__(
        self,
        radius: float,
        outwards_normal: np.ndarray,  # Pointing from the origin of the sphere to the mirror's center.
        center: Optional[np.ndarray] = None,  # Not the center of the sphere but the center of
        # the plate, where the beam should hit.
        origin: Optional[np.ndarray] = None,  # The center of the sphere.
        curvature_sign: int = 1,
        name: Optional[str] = None,
        thermal_properties: Optional[MaterialProperties] = None,
    ):

        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            material_properties=thermal_properties,
            radius=radius,
            center=center,
            origin=origin,
            curvature_sign=curvature_sign,
        )

    def reflect_direction_exact(self, ray: Ray, intersection_point: Optional[np.ndarray] = None) -> np.ndarray:
        # Notice that this function does not reflect along the normal of the mirror but along the normal projection
        # of the ray on the mirror.
        if intersection_point is None:
            intersection_point = self.find_intersection_with_ray(ray)
        mirror_normal_vector = (self.origin - intersection_point) * self.curvature_sign  # m_rays | 3
        mirror_normal_vector = normalize_vector(mirror_normal_vector)
        dot_product = np.sum(ray.k_vector * mirror_normal_vector, axis=-1)  # m_rays  # This dot product is written
        # like so because both tensors have the same shape and the dot product is calculated along the last axis.
        # you could also perform this product by transposing the second tensor and then dot multiplying the two tensors,
        # but this it would be cumbersome to do so.
        reflected_direction_vector = (
            ray.k_vector - 2 * dot_product[..., np.newaxis] * mirror_normal_vector
        )  # m_rays | 3
        return reflected_direction_vector

    def reflect_direction_paraxial(self, ray: Ray) -> np.ndarray:
        # This is maybe wrong but does not matter too much because anyway they are not used for the central line finding
        intersection_point = self.find_intersection_with_ray(ray, paraxial=True)
        return self.reflect_direction_exact(ray, intersection_point=intersection_point)

    def ABCD_matrix(self, cos_theta_incoming: float = None):
        # order of rows/columns elements is [theta, theta, phi, phi]
        # An approximation is done here (beyond the small angles' approximation) by assuming that the central line
        # lives in the x,y plane, such that the plane of incidence is the x,y plane (parameterized by phi and phi)
        # and the sagittal plane is its transverse (parameterized by theta and theta).
        # This is justified for small perturbations of a cavity whose central line actually lives in the x,y plane.
        # It is not really justified for bigger perturbations and should be corrected.
        # It should be corrected by first finding the real axes, # And then apply a rotation matrix to this matrix on
        # both sides.
        ABCD = np.array(
            [
                [1, 0, 0, 0],
                [-2 * cos_theta_incoming / self.radius, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 2 / (self.radius * cos_theta_incoming), -1],
            ]
        )
        return ABCD

    def plot_2d(self, ax: Optional[plt.Axes] = None, name: Optional[str] = None):
        if ax is None:
            fig, ax = plt.subplots()
        d_theta = 0.3
        p = np.linspace(-d_theta, d_theta, 50)
        p_grey = np.linspace(d_theta, -d_theta + 2 * np.pi, 100)
        points = self.parameterization(0, p)
        grey_points = self.parameterization(0, p_grey)
        ax.plot(points[:, 0], points[:, 1], "b-")
        ax.plot(
            grey_points[:, 0],
            grey_points[:, 1],
            color=(0.81, 0.81, 0.81),
            linestyle="-.",
            linewidth=0.5,
            label=None,
        )
        ax.plot(self.origin[0], self.origin[1], "bo")

    def thermal_transformation(
        self,
        P_laser_power: float,
        w_spot_size: float,
        transform_mirror: bool = True,
        **kwargs,
    ):
        if not transform_mirror or np.isnan(w_spot_size):
            return self
        else:
            poisson_ratio_factor = (1 + self.material_properties.nu_poisson_ratio) / (
                1 - self.material_properties.nu_poisson_ratio
            )
            delta_T = (
                PHYSICAL_SIZES_DICT["c_mirror_radius_expansion"]
                * P_laser_power
                * self.material_properties.beta_surface_absorption
                / (self.material_properties.kappa_conductivity * w_spot_size)
            )
            delta_curvature = (
                -delta_T * self.material_properties.alpha_expansion * poisson_ratio_factor / w_spot_size
            )  # The minus is because we are cooling it down.
            # delta_z = delta_curvature * w_spot_size ** 2  # Technically the curvature is calculated based on this delta_z, but I skip it in the code and calculate the curvature directly.
            new_radius = (self.radius**-1 + delta_curvature) ** -1  # ARBITRARY - TAKING ONLY THE T AXIS
            self.material_properties.temperature = ROOM_TEMPERATURE - delta_T  # The delta_T is negative, and after
            # cooling the mirror goes to room temperature. Therefore, the temperature is when heated is the room
            # temperature minus the delta_T.

            new_thermal_properties = copy.copy(self.material_properties)
            new_thermal_properties.temperature = delta_T

            new_mirror = CurvedMirror(
                radius=new_radius,
                outwards_normal=self.outwards_normal,
                center=self.center,
                thermal_properties=new_thermal_properties,
            )
            new_mirror.radius = new_radius
            return new_mirror


class CurvedRefractiveSurface(CurvedSurface, PhysicalSurface):
    def __init__(
        self,
        radius: float,
        outwards_normal: np.ndarray,  # Pointing from the origin of the sphere to the mirror's center.
        center: Optional[np.ndarray] = None,  # Not the center of the sphere but the center of the plate.
        origin: Optional[np.ndarray] = None,  # The center of the sphere.
        n_1: float = 1,
        n_2: float = 1.5,
        curvature_sign: int = 1,
        name: Optional[str] = None,
        thermal_properties: Optional[MaterialProperties] = None,
        thickness: Optional[float] = 5e-4,
    ):
        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            material_properties=thermal_properties,
            radius=radius,
            center=center,
            origin=origin,
            curvature_sign=curvature_sign,
        )
        self.n_1 = n_1
        self.n_2 = n_2
        self.thickness = thickness

    def reflect_direction_exact(self, ray: Ray, intersection_point: Optional[np.ndarray] = None) -> np.ndarray:
        # explanable derivation of the calculation in lab archives: https://mynotebook.labarchives.com/MTM3NjE3My41fDEwNTg1OTUvMTA1ODU5NS9Ob3RlYm9vay8zMjQzMzA0MzY1fDM0OTMzNjMuNQ==/page/11290221-33
        if intersection_point is None:
            intersection_point = self.find_intersection_with_ray(ray)
        n_backwards = (
            self.origin - intersection_point
        ) * self.curvature_sign  # m_rays | 3  # The normal to the surface
        n_backwards = normalize_vector(n_backwards)
        n_forwards = -n_backwards
        cos_theta_incoming = np.clip(np.sum(ray.k_vector * n_forwards, axis=-1), a_min=-1, a_max=1)  # m_rays
        n_orthogonal = (
            ray.k_vector - cos_theta_incoming[..., np.newaxis] * n_forwards
        )  # m_rays | 3  # This is the vector that is orthogonal to the normal to the surface and lives in the plane spanned by the ray and the normal to the surface (grahm-schmidt process).
        n_orthogonal_norm = np.linalg.norm(n_orthogonal, axis=-1)  # m_rays
        if isinstance(n_orthogonal_norm, float) and n_orthogonal_norm < 1e-15:
            reflected_direction_vector = n_forwards
        else:
            preactically_normal_incidences = n_orthogonal_norm < 1e-15
            n_orthogonal[preactically_normal_incidences] = (
                np.nan
            )  # This is done so that the normalization does not throw an error.
            n_orthogonal = normalize_vector(n_orthogonal)
            sin_theta_outgoing = np.sqrt((self.n_1 / self.n_2) ** 2 * (1 - cos_theta_incoming**2))  # m_rays
            reflected_direction_vector = (
                n_forwards * stable_sqrt(1 - sin_theta_outgoing[..., np.newaxis] ** 2)
                + n_orthogonal * sin_theta_outgoing[..., np.newaxis]
            )  # m_rays | 3
            reflected_direction_vector[preactically_normal_incidences] = n_forwards[
                preactically_normal_incidences
            ]  # For the nans we initiated before, we just want the normal to the surface to be the new direction of the ray
        return reflected_direction_vector

    def ABCD_matrix(self, cos_theta_incoming: float = None) -> np.ndarray:
        cos_theta_outgoing = np.sqrt(1 - (self.n_1 / self.n_2) ** 2 * (1 - cos_theta_incoming**2))
        R_signed = self.radius * self.curvature_sign
        delta_n_e_out_of_plane = self.n_2 * cos_theta_outgoing - self.n_1 * cos_theta_incoming
        delta_n_e_in_plane = delta_n_e_out_of_plane / (cos_theta_incoming * cos_theta_outgoing)

        # See the comment in the ABCD_matrix method of the CurvedSurface class for an explanation of the approximation.
        ABCD = np.array(
            [
                [1, 0, 0, 0],  # theta
                [delta_n_e_out_of_plane / (R_signed * self.n_2), self.n_1 / self.n_2, 0, 0],  # theta
                [0, 0, cos_theta_outgoing / cos_theta_incoming, 0],  # phi
                [
                    0,
                    0,
                    delta_n_e_in_plane / (R_signed * self.n_2),
                    cos_theta_incoming * self.n_1 / (cos_theta_outgoing * self.n_2),
                ],
            ]
        )  # phi
        return ABCD

    def thermal_transformation(
        self,
        P_laser_power: float,
        w_spot_size: float,
        n_surface_transform_lens: bool = True,
        n_volumetric_transform_lens: bool = True,
        curvature_transform_lens: bool = True,
        change_lens_by_changing_n: bool = True,
        change_lens_by_changing_R: bool = False,
        z_transform_lens: bool = False,
        **kwargs,
    ):
        if np.isnan(w_spot_size):
            return self
        n_inside = np.max((self.n_1, self.n_2))
        delta_T_volumetric = (
            PHYSICAL_SIZES_DICT["c_lens_volumetric_absorption"]
            * self.material_properties.alpha_volume_absorption
            * P_laser_power
            / self.material_properties.kappa_conductivity
        )  # ARBITRARY - CHANGE THE DIMENSIONLESS CONSTANT
        delta_T_surface = (
            PHYSICAL_SIZES_DICT["c_lens_focal_length_expansion"]
            * self.material_properties.beta_surface_absorption
            * P_laser_power
            / (self.material_properties.kappa_conductivity * w_spot_size)
        )  # ARBITRARY - CHANGE THE DIMENSIONLESS CONSTANT
        delta_T = delta_T_volumetric + delta_T_surface
        self.material_properties.temperature = ROOM_TEMPERATURE - delta_T

        common_coefficient = (
            self.material_properties.beta_surface_absorption
            * P_laser_power
            / (self.material_properties.kappa_conductivity * w_spot_size**2)
        )
        delta_optical_length_curvature_n_surface = (
            -PHYSICAL_SIZES_DICT["c_lens_focal_length_expansion"] * common_coefficient * self.material_properties.dn_dT
        )
        delta_optical_length_curvature_n_volumetric = (
            -PHYSICAL_SIZES_DICT["c_lens_volumetric_absorption"]
            * self.material_properties.alpha_volume_absorption
            * P_laser_power
            * self.material_properties.dn_dT
            / self.material_properties.kappa_conductivity
            * (1 / self.radius + self.thickness / w_spot_size**2)
        )  # (1 / self.radius self.thickness / w_spot_size ** 2) The last parenthesis should be this but the 1/R is negligible.
        delta_optical_length_curvature_buldging = (
            -PHYSICAL_SIZES_DICT["c_lens_focal_length_expansion"]
            * common_coefficient
            * self.material_properties.alpha_expansion
            * n_inside
            * (1 + self.material_properties.nu_poisson_ratio)
            / (1 - self.material_properties.nu_poisson_ratio)
        )

        delta_optical_length_curvature = (
            delta_optical_length_curvature_n_surface * n_surface_transform_lens
            + delta_optical_length_curvature_n_volumetric * n_volumetric_transform_lens
            + delta_optical_length_curvature_buldging * curvature_transform_lens
        )

        # A way which is also correct but less readable and less intuitive:
        # delta_optical_length_curvature_z = common_coefficient * n_inside * self.material_properties.alpha_expansion * (1+self.material_properties.nu_poisson_ratio) / (1-self.material_properties.nu_poisson_ratio)
        # radius_new = self.radius * n_inside / (n_inside + delta_optical_length_curvature_z * self.radius)

        if change_lens_by_changing_n:
            radius_new = self.radius
            n_new = n_inside - delta_optical_length_curvature * self.radius

        elif change_lens_by_changing_R:
            radius_new = n_inside * self.radius / (n_inside - delta_optical_length_curvature * self.radius)
            n_new = n_inside
        else:
            raise ValueError("at least change_lens_by_changing_n or change_lens_by_changing_R has to be True")

        if self.n_1 == 1:
            n_1 = 1
            n_2 = n_new
        else:
            n_1 = n_new
            n_2 = 1

        if z_transform_lens:
            # delta_z = 0
            # center_new = self.center + delta_z * self.outwards_normal
            raise NotImplementedError
        else:
            center_new = self.center

        new_thermal_properties = copy.copy(self.material_properties)
        new_thermal_properties.temperature = ROOM_TEMPERATURE

        return CurvedRefractiveSurface(
            radius=radius_new,
            outwards_normal=self.outwards_normal,
            center=center_new,
            n_1=n_1,
            n_2=n_2,
            curvature_sign=self.curvature_sign,
            name=self.name,
            thermal_properties=new_thermal_properties,
        )
        # return self


def generate_lens_from_params(params: OpticalElementParams, name: Optional[str] = "Lens"):
    p = params
    # generates a convex-convex lens from the parameters
    center = np.array([p.x, p.y, p.z])
    forward_direction = unit_vector_of_angles(p.theta, p.phi)

    # Generate names according to the direction the lens is pointing to
    main_axis = np.argmax(np.abs(forward_direction))
    directions_nams = [["_left", "_right"], ["_down", "_up"], ["_back", "_front"]]
    suffixes = directions_nams[main_axis]
    if forward_direction[main_axis] < 0:
        suffixes = suffixes[::-1]
    if name is None:
        name = "Lens"
    names = [name + suffix for suffix in suffixes]

    center_1 = center - (1 / 2) * p.T_c * forward_direction
    center_2 = center + (1 / 2) * p.T_c * forward_direction
    surface_1 = CurvedRefractiveSurface(
        radius=p.r_1,
        outwards_normal=-forward_direction,
        center=center_1,
        n_1=p.n_outside_or_before,
        n_2=p.n_inside_or_after,
        curvature_sign=-1,
        name=names[0],
        thermal_properties=p.material_properties,
        thickness=p.T_c / 2,
    )

    surface_2 = CurvedRefractiveSurface(
        radius=p.r_2,
        outwards_normal=forward_direction,
        center=center_2,
        n_1=p.n_inside_or_after,
        n_2=p.n_outside_or_before,
        curvature_sign=1,
        name=names[1],
        thermal_properties=p.material_properties,
        thickness=p.T_c / 2,
    )
    return surface_1, surface_2


def convert_curved_refractive_surface_to_ideal_lens(surface: CurvedRefractiveSurface):
    focal_length = 1 / (surface.n_2 - surface.n_1) * surface.radius * (-1 * surface.curvature_sign)
    ideal_lens = IdealLens(
        outwards_normal=surface.outwards_normal,
        center=surface.center,
        focal_length=focal_length,
        name=surface.name,
        thermal_properties=surface.material_properties,
    )

    flat_refractive_surface = FlatRefractiveSurface(
        outwards_normal=surface.outwards_normal,
        center=surface.center,
        n_1=surface.n_1,
        n_2=surface.n_2,
        name=surface.name + "_refractive_surface",
        thermal_properties=surface.material_properties,
    )

    return ideal_lens, flat_refractive_surface


def generate_thick_ideal_lens_from_params(params: OpticalElementParams, name: Optional[str] = "Lens"):
    surface_1, surface_4 = generate_lens_from_params(params, name)
    ideal_lens_1 = convert_curved_refractive_surface_to_ideal_lens(surface_1)
    ideal_lens_2 = convert_curved_refractive_surface_to_ideal_lens(surface_4)
    return ideal_lens_1, ideal_lens_2


class Arm:
    def __init__(
        self,
        surface_0: Surface,
        surface_1: Surface,
        central_line: Optional[Ray] = None,
        mode_parameters_on_surface_0: Optional[LocalModeParameters] = None,
        mode_parameters_on_surface_1: Optional[LocalModeParameters] = None,
        mode_principle_axes: Optional[np.ndarray] = None,
        lambda_0_laser: Optional[float] = None,
    ):
        self.surface_0 = surface_0
        self.surface_1 = surface_1
        self.mode_parameters_on_surface_0 = mode_parameters_on_surface_0
        self.mode_parameters_on_surface_1 = mode_parameters_on_surface_1
        self.central_line = central_line
        self.mode_principle_axes = mode_principle_axes
        self.lambda_0_laser = lambda_0_laser
        if isinstance(surface_0, CurvedRefractiveSurface):
            self.n = surface_0.n_2
        elif isinstance(surface_1, CurvedRefractiveSurface):
            self.n = surface_1.n_1
        else:
            self.n = 1
        if isinstance(surface_0, CurvedRefractiveSurface) and isinstance(surface_1, CurvedRefractiveSurface):
            assert surface_0.n_2 == surface_1.n_1

    def propagate(self, ray: Ray, use_paraxial_ray_tracing: bool = False):

        if isinstance(self.surface_1, PhysicalSurface):
            ray = self.surface_1.reflect_ray(ray, paraxial=use_paraxial_ray_tracing)
        else:
            new_position = self.surface_1.find_intersection_with_ray(ray, paraxial=use_paraxial_ray_tracing)
            ray = Ray(new_position, ray.k_vector)
        return ray

    @property
    def ABCD_matrix_free_space(self):
        if self.central_line is None:
            raise ValueError("Central line not set")
        matrix = ABCD_free_space(self.central_line.length)
        return matrix

    @property
    def ABCD_matrix_reflection(self):
        if self.central_line is None:
            raise ValueError("Central line not set")
        cos_theta = np.abs(self.central_line.k_vector @ self.surface_1.outwards_normal)  # ABS because we want the
        # angle between the ray and the normal to be positive
        if isinstance(self.surface_1, PhysicalSurface):
            matrix = self.surface_1.ABCD_matrix(cos_theta)
        else:
            matrix = np.eye(4)
        return matrix

    @property
    def ABCD_matrix(self):
        matrix = self.ABCD_matrix_reflection @ self.ABCD_matrix_free_space
        return matrix

    def propagate_local_mode_parameters(self):
        if self.mode_parameters_on_surface_0 is None:
            raise ValueError("Mode parameters on surface 1 not set")
        self.mode_parameters_on_surface_1 = propagate_local_mode_parameter_through_ABCD(
            self.mode_parameters_on_surface_0, self.ABCD_matrix_free_space, n_1=self.n, n_2=self.n
        )
        next_mode_parameters = propagate_local_mode_parameter_through_ABCD(
            self.mode_parameters_on_surface_1,
            self.ABCD_matrix_reflection,
            n_1=self.surface_1.to_params.n_outside_or_before,
            n_2=self.surface_1.to_params.n_inside_or_after,
        )
        return next_mode_parameters

    # @property
    # def mode_principle_axes(self):
    #     if self.central_line is None:
    #         raise ValueError('Central line not set')
    #     z_hat = np.array([0, 0, 1])
    #     pseudo_x = np.cross(z_hat, self.central_line.k_vector)
    #     mode_principle_axes = np.stack([z_hat, pseudo_x], axis=-1).T  # [z_x, z_y, z_z], [x_x, x_y, x_z]
    #     return mode_principle_axes

    @property
    def mode_parameters(self):
        if self.mode_parameters_on_surface_0 is None:
            return None
        center = (
            self.central_line.origin
            - self.mode_parameters_on_surface_0.z_minus_z_0[..., np.newaxis] / self.n * self.central_line.k_vector
        )
        mode_parameters = ModeParameters(
            center=center,
            k_vector=self.central_line.k_vector,
            w_0=self.mode_parameters_on_surface_0.w_0,
            principle_axes=self.mode_principle_axes,
            lambda_0_laser=self.lambda_0_laser,
            n=self.n,
        )
        return mode_parameters

    def local_mode_parameters_on_a_point(self, point: np.ndarray):
        if self.central_line is None:
            raise ValueError("Central line not set")
        if self.mode_parameters_on_surface_0 is None:
            return None

        point_plane_distance_from_surface_1 = (point - self.central_line.origin) @ self.central_line.k_vector
        propagation_ABCD = ABCD_free_space(point_plane_distance_from_surface_1)
        local_mode_parameters = propagate_local_mode_parameter_through_ABCD(
            self.mode_parameters_on_surface_0, propagation_ABCD, n_1=self.n, n_2=self.n
        )
        return local_mode_parameters

    @property
    def surfaces(self):
        return [self.surface_0, self.surface_1]

    @property
    def mode_parameters_on_surfaces(self):
        return [self.mode_parameters_on_surface_0, self.mode_parameters_on_surface_1]

    def calculate_incidence_angle(self, surface_index: int) -> float:

        return calculate_incidence_angle(
            surface=self.surfaces[surface_index],
            mode_parameters=self.mode_parameters,
        )

    def specs(self):
        list_of_data_frames = []
        for i in [0, 1]:
            local_mode_parameters = self.mode_parameters_on_surfaces[i]
            spot_size_on_surface = local_mode_parameters.spot_size[0]
            surface = self.surfaces[i]
            if isinstance(surface, CurvedRefractiveSurface):
                if i == 0 and surface.n_2 == 1 or i == 1 and surface.n_1 == 1:
                    angle_side = ""
                else:
                    angle_side = "_inside"
            else:
                angle_side = ""
            if local_mode_parameters.spot_size[0] != local_mode_parameters.spot_size[1]:
                warnings.warn("Not yep implemented for astigmatic systems! using the spot size for one arbitrary axis")

            angle_of_incidence_deg = calculate_incidence_angle(self.surfaces[i], self.mode_parameters)

            df = pd.DataFrame(
                {
                    "Element": [surface.name] * 4,
                    "Parameter": [
                        "Spot size diameter [m]",
                        "Minimal clear aperture diameter [m]",
                        "Temperature raise [K]",
                        f"Angle of incidence{angle_side} [deg]",
                    ],
                    "Value": [
                        spot_size_on_surface * 2,
                        spot_size_on_surface * 2 * 2.5,
                        surface.material_properties.temperature - ROOM_TEMPERATURE,
                        angle_of_incidence_deg,
                    ],
                },
            )

            list_of_data_frames.append(df)
        joined_df = pd.concat(list_of_data_frames)
        return joined_df


class Cavity:
    def __init__(
        self,
        physical_surfaces: List[PhysicalSurface],
        standing_wave: bool = False,
        lambda_0_laser: Optional[float] = None,
        params: Optional[List[OpticalElementParams]] = None,
        names: Optional[List[str]] = None,
        set_central_line: bool = True,
        set_mode_parameters: bool = True,
        set_initial_surface: bool = False,
        t_is_trivial: bool = False,
        p_is_trivial: bool = False,
        power: Optional[float] = None,
        initial_local_mode_parameters: Optional[LocalModeParameters] = None,
        initial_mode_parameters: Optional[ModeParameters] = None,
        use_brute_force_for_central_line: bool = False,  # remove it once we know it works
        debug_printing_level: int = 0,  # 0 for no prints, 1 for main prints, 2 for all prints
        use_paraxial_ray_tracing: bool = False,
    ):
        self.standing_wave = standing_wave
        self.physical_surfaces = physical_surfaces
        self.arms: List[Arm] = [
            Arm(
                self.physical_surfaces_ordered[i],
                self.physical_surfaces_ordered[np.mod(i + 1, len(self.physical_surfaces_ordered))],
                lambda_0_laser=lambda_0_laser,
            )
            for i in range(len(self.physical_surfaces_ordered))
        ]
        self.central_line_successfully_traced: Optional[bool] = None
        self.resonating_mode_successfully_traced: Optional[bool] = None
        self.lambda_0_laser: Optional[float] = lambda_0_laser
        self.params = params
        self.names_memory = names
        self.t_is_trivial = t_is_trivial
        self.p_is_trivial = p_is_trivial
        self.power = power
        self.use_brute_force_for_central_line = use_brute_force_for_central_line
        self.debug_printing_level = debug_printing_level
        self.use_paraxial_ray_tracing = use_paraxial_ray_tracing

        if set_central_line:
            self.set_central_line()
        if set_mode_parameters:
            self.set_mode_parameters(
                mode_parameters_first_arm=initial_mode_parameters,
                local_mode_parameters_first_surface=initial_local_mode_parameters,
            )
        if set_initial_surface:
            self.set_initial_surface()

    @staticmethod
    def from_params(
        params: Union[np.ndarray, List[OpticalElementParams]],
        names: Optional[List[str]] = None,
        **kwargs,
    ):
        if isinstance(params, np.ndarray):
            p = [OpticalElementParams.from_array(params[i, :]) for i in range(len(params))]
        else:
            p = params
        physical_surfaces = []
        if names is None:
            names = [f"surface_{i}" for i in range(len(p))]
        for i in range(len(p)):
            surface_temp = Surface.from_params(p[i], name=names[i])
            if isinstance(surface_temp, tuple):
                physical_surfaces.extend(surface_temp)
            else:
                physical_surfaces.append(surface_temp)
        cavity = Cavity(
            physical_surfaces,
            params=p,
            names=names,
            **kwargs,
        )
        return cavity

    @property
    def to_params(self) -> List[OpticalElementParams]:
        if self.params is None:
            params = [surface.to_params for surface in self.physical_surfaces]
        else:
            params = self.params
        return params

    @property
    def to_array(self) -> np.ndarray:
        array = np.stack([param.to_array for param in self.to_params], axis=0)
        return array

    @property
    def id(self):
        hashed_str = int(md5(self.params).hexdigest()[:5], 16)
        return hashed_str

    @property
    def physical_surfaces_ordered(self):
        if self.standing_wave:
            backwards_list = copy.deepcopy(self.physical_surfaces[-2:0:-1])
            for surface in backwards_list:
                if isinstance(surface, CurvedRefractiveSurface):
                    surface.curvature_sign = -surface.curvature_sign
                    n_1, n_2 = surface.n_1, surface.n_2
                    surface.n_1 = n_2
                    surface.n_2 = n_1
            return self.physical_surfaces + backwards_list
        else:
            return self.physical_surfaces

    @property
    def central_line(self):
        if self.arms[0].central_line is None:
            return None
        else:
            return [arm.central_line for arm in self.arms]

    @property
    def ABCD_matrices(self):
        if self.arms[0].central_line is None:
            return None
        else:
            ABCD_list = [arm.ABCD_matrix for arm in self.arms]
            return ABCD_list

    @property
    def ABCD_round_trip(self):
        if self.arms[0].central_line is None:
            return None
        else:
            return np.linalg.multi_dot(self.ABCD_matrices[::-1])

    @property
    def mode_parameters(self):
        if self.arms[0].central_line is None:
            return None
        else:
            return [arm.mode_parameters for arm in self.arms]

    @property
    def surfaces(self):
        return [arm.surface_0 for arm in self.arms]

    @property
    def default_initial_k_vector(self) -> np.ndarray:
        if self.central_line is not None and self.central_line_successfully_traced:
            initial_k_vector = self.central_line[0].k_vector
        else:
            initial_k_vector = self.arms[0].surface_1.center - self.arms[0].surface_0.center
            initial_k_vector = normalize_vector(initial_k_vector)
        return initial_k_vector

    @property
    def default_initial_angles(self) -> Tuple[float, float]:
        initial_k_vector = self.default_initial_k_vector
        theta, phi = angles_of_unit_vector(initial_k_vector)
        return theta, phi

    @property
    def default_initial_ray(self) -> Ray:
        if self.central_line_successfully_traced:
            return self.central_line[0]
        else:
            initial_k_vector = self.default_initial_k_vector
            initial_ray = Ray(origin=self.arms[0].surface_0.center, k_vector=initial_k_vector)
            return initial_ray

    @property
    def names(self):
        if self.names_memory is None:
            return [surface.name for surface in self.physical_surfaces]
        else:
            return self.names_memory

    @property
    def perturbable_params_names(self):
        perturbable_params_names_list = params_to_perturbable_params_names(
            self.params, self.t_is_trivial and self.p_is_trivial
        )
        return perturbable_params_names_list

    @property
    def roundtrip_power_losses(self):
        # if roundtrip_power_losses = 0.2 then every roundtrip 0.2 of the power is lost
        if self.central_line_successfully_traced is False:
            return None
        # losses = 0
        starting_power = 1
        for arm in self.arms:
            first_surface = arm.surface_0
            if isinstance(first_surface, (CurvedMirror, FlatMirror)):
                surface_unlost_portion = first_surface.material_properties.intensity_reflectivity
            elif isinstance(first_surface, CurvedRefractiveSurface):
                surface_unlost_portion = first_surface.material_properties.intensity_transmittance
            else:
                raise ValueError(f"Surface type {type(first_surface)} not implemented in this function")
            alpha = 0
            if hasattr(
                first_surface, "n_2"
            ):  # Do not include volumetric losses if the arms is made of air. this is a bad implementation, and the volumetric losses should be included in the arms properties.
                if first_surface.n_2 != 1:
                    alpha = first_surface.material_properties.alpha_volume_absorption
            volume_absorption_unlost_portion_log = alpha * arm.central_line.length
            volume_absorption_unlost_portion = np.exp(-volume_absorption_unlost_portion_log)
            if isinstance(first_surface, (CurvedMirror, FlatMirror)):
                starting_power *= surface_unlost_portion
            elif isinstance(first_surface, CurvedRefractiveSurface):
                starting_power *= surface_unlost_portion * volume_absorption_unlost_portion
            # losses += surface_coherent_loss + surface_absorption_loss + volume_absorption_loss_log
        return 1 - starting_power  # , losses

    @property
    def roundtrip_optical_length(self):
        if self.central_line_successfully_traced is False:
            return None
        else:
            optical_length = 0
            for arm in self.arms:
                if isinstance(arm.surface_0, CurvedRefractiveSurface):
                    optical_length += arm.central_line.length * arm.surface_0.n_2
                else:
                    optical_length += arm.central_line.length
        return optical_length

    @property
    def roundtrip_time(self):
        if self.central_line_successfully_traced is False:
            return None
        else:
            return self.roundtrip_optical_length / C_LIGHT_SPEED

    @property
    def free_spectral_range(self):
        if self.central_line_successfully_traced is False:
            return None
        else:
            return 1 / self.roundtrip_time

    @property
    def power_decay_rate(self):
        if self.central_line_successfully_traced is False:
            return None
        else:
            return self.roundtrip_power_losses / self.roundtrip_time

    @property
    def amplitude_decay_rate(self):
        if self.central_line_successfully_traced is False:
            return None
        else:
            return self.power_decay_rate / 2

    @property
    def finesse(self):
        return np.pi * self.free_spectral_range / self.amplitude_decay_rate

    def trace_ray(self, ray: Ray) -> List[Ray]:
        ray_history = [ray]
        for arm in self.arms:
            ray = arm.propagate(ray, use_paraxial_ray_tracing=self.use_paraxial_ray_tracing)
            ray_history.append(ray)
        return ray_history

    def trace_ray_parametric(self, starting_position_and_angles: np.ndarray) -> Tuple[np.ndarray, List[Ray]]:
        # Like trace ray, but works as a function of the starting position and angles as parameters on the starting
        # surface, instead of the starting position and angles as a vector in 3D space.

        initial_ray = self.ray_of_initial_parameters(starting_position_and_angles)
        ray_history = self.trace_ray(initial_ray)
        final_intersection_point = ray_history[-1].origin
        t_o, p_o = self.arms[0].surface_0.get_parameterization(final_intersection_point)  # Here it is the initial
        # surface on purpose: the final ray's origin should be on the initial surface, after one round trip.
        theta_o, phi_o = angles_of_unit_vector(ray_history[-1].k_vector)
        final_position_and_angles = np.stack([t_o, theta_o, p_o, phi_o], axis=-1)
        return final_position_and_angles, ray_history

    def f_roots(self, starting_position_and_angles: np.ndarray) -> np.ndarray:
        # The roots of this function are the initial parameters for the central line. (position x, y, angles theta, phi)
        # try:
        final_position_and_angles, _ = self.trace_ray_parametric(starting_position_and_angles / STRETCH_FACTOR)
        diff = np.zeros_like(starting_position_and_angles)
        diff[..., [0, 2]] = (
            final_position_and_angles[..., [0, 2]] - starting_position_and_angles[..., [0, 2]] / STRETCH_FACTOR
        )
        diff[..., [1, 3]] = angles_difference(
            starting_position_and_angles[..., [1, 3]] / STRETCH_FACTOR,
            final_position_and_angles[..., [1, 3]],
        )
        diff[np.isnan(diff)] = np.inf
        return diff * STRETCH_FACTOR

    def f_roots_standing_wave(self, angles: np.ndarray):
        # This function returns 0 also if the ray is pointing to the exact opposite direction of the central line
        # make sure it does not create problems.
        last_ray_index = len(self.physical_surfaces) - 2  # minus one for the first surface and -1 because of python's
        # 0 indexing.
        k_vector = unit_vector_of_angles(angles[0], angles[1])
        ray = Ray(self.physical_surfaces[0].origin, k_vector)
        ray_history = self.trace_ray(ray)
        last_arms_ray = ray_history[last_ray_index]  # -2

        origins_plane = FlatSurface(
            outwards_normal=self.physical_surfaces[-1].outwards_normal, center=self.physical_surfaces[-1].origin
        )
        intersection_point = origins_plane.find_intersection_with_ray(
            last_arms_ray, paraxial=self.use_paraxial_ray_tracing
        )
        t, p = origins_plane.get_parameterization(intersection_point)

        # Alternative syntax (that results in rotated parameterization)
        # d = np.cross(last_arms_ray.k_vector, rays_origin_to_surface_origin)  # This is a signed distance, which
        # transverse_spanning_vector_1, transverse_spanning_vector_2 = self.physical_surfaces[-1].spanning_vectors()  # -1
        # d_1 = d @ transverse_spanning_vector_1
        # d_2 = d @ transverse_spanning_vector_2
        # is good for the solver.
        # print(angles[1] - np.pi, phi)
        result_array = np.array([t, p])
        return result_array

    def find_central_line_solver(self):
        theta_initial_guess, phi_initial_guess = self.default_initial_angles
        initial_guess = np.array([0, theta_initial_guess, 0, phi_initial_guess]) * STRETCH_FACTOR

        if self.t_is_trivial and self.p_is_trivial:
            central_line_initial_parameters = initial_guess
        else:
            if self.t_is_trivial and not self.p_is_trivial:
                initial_guess_subspace = initial_guess[[2, 3]]
                f_roots_subspace = lambda x: self.f_roots(np.array([initial_guess[0], initial_guess[1], x[0], x[1]]))[
                    [2, 3]
                ]
                central_line_initial_parameters: np.ndarray = optimize.fsolve(f_roots_subspace, initial_guess_subspace)
                central_line_initial_parameters = np.concatenate(
                    (initial_guess[[0, 1]], central_line_initial_parameters)
                )
            elif not self.t_is_trivial and self.p_is_trivial:
                initial_guess_subspace = initial_guess[[0, 1]]
                f_roots_subspace = lambda x: self.f_roots(np.array([x[0], x[1], initial_guess[2], initial_guess[3]]))[
                    [0, 1]
                ]
                central_line_initial_parameters: np.ndarray = optimize.fsolve(f_roots_subspace, initial_guess_subspace)
                central_line_initial_parameters = np.concatenate(
                    (central_line_initial_parameters, initial_guess[[2, 3]])
                )
            else:
                central_line_initial_parameters: np.ndarray = optimize.fsolve(self.f_roots, initial_guess)
            # In the documentation it says optimize.fsolve returns a solution, together with some flags, and also this
            # is how pycharm suggests to use it. But in practice it returns only the solution, not sure why.

        root_error = np.linalg.norm(self.f_roots(central_line_initial_parameters))
        central_line_initial_parameters /= STRETCH_FACTOR

        print(f"root_error: {root_error}")
        print(f"diff: {self.f_roots(central_line_initial_parameters)}")

        central_line_successfully_traced = root_error < CENTRAL_LINE_TOLERANCE * STRETCH_FACTOR

        return central_line_initial_parameters, central_line_successfully_traced

    def find_central_line_brute_force(
        self,
        N_resolution: int = 11,
        range_limit: float = 1e-4,
        zoom_factor: float = 1.4,
        N_iterations: int = 50,
    ) -> Tuple[np.ndarray, bool]:
        theta_initial_guess, phi_initial_guess = self.default_initial_angles
        central_line_initial_parameters = np.array([0, theta_initial_guess, 0, phi_initial_guess])

        if self.t_is_trivial and self.p_is_trivial:
            return central_line_initial_parameters, True

        if self.debug_printing_level >= 2:
            fig, ax = plt.subplots(N_iterations, 3, figsize=(24, N_iterations * 3))

        for i in range(N_iterations):
            initial_parameters = generate_initial_parameters_grid(
                central_line_initial_parameters,
                range_limit,
                N_resolution,
                p_is_trivial=self.p_is_trivial,
                t_is_trivial=self.t_is_trivial,
            )
            diff = self.f_roots(initial_parameters)
            diff_norm = np.linalg.norm(diff, axis=-1)

            smallest_elements_index = np.unravel_index(np.argmin(diff_norm), diff.shape[:-1])
            central_line_initial_parameters = initial_parameters[smallest_elements_index]
            range_limit /= zoom_factor

            central_line_successfully_traced = diff_norm[smallest_elements_index] < CENTRAL_LINE_TOLERANCE
            if self.debug_printing_level >= 2:
                print(f"iteration {i}, error: {diff_norm[smallest_elements_index]}")
                print(f"iteration {i}, center: {central_line_initial_parameters}\n")
                print(f"iteration {i+1}, range_limit: {range_limit:.3e}")
                ax[i, 0].imshow(diff_norm)
                # plt.colorbar()
                # Add a dot at the minimum:
                ax[i, 0].scatter(smallest_elements_index[1], smallest_elements_index[0], color="r")
                if self.p_is_trivial:
                    parameters_indices = [0, 1]
                else:
                    parameters_indices = [2, 3]
                diff_position = diff_norm[:, smallest_elements_index[1]]
                ax[i, 1].plot(initial_parameters[:, 0, parameters_indices[0]], diff_position)
                ax[i, 1].axvline(initial_parameters[smallest_elements_index[0], 0, parameters_indices[0]], color="r")
                ax[i, 1].set_title(f"{i}: position")

                diff_angle = diff_norm[smallest_elements_index[0], :]
                ax[i, 2].plot(initial_parameters[0, :, parameters_indices[1]], diff_angle)
                ax[i, 2].axvline(initial_parameters[0, smallest_elements_index[1], parameters_indices[1]], color="r")
                ax[i, 2].set_title(f"{i}: angle")

        if self.debug_printing_level >= 2:
            plt.show()
        if self.debug_printing_level >= 1:
            print(f"root_error: {diff_norm[smallest_elements_index]}")
            print(f"diff: {diff[smallest_elements_index]}")

        return central_line_initial_parameters, central_line_successfully_traced

    def find_central_line_standing_wave(self):
        # This function assumes the centers of the origins (sphere's center) of the first and last mirrors are withing
        # their arms, which will not be for the case of the astigmatic cavity with the extra mirror, but for now it should work.
        if not isinstance(self.physical_surfaces[0], CurvedMirror) and isinstance(
            self.physical_surfaces[-1], CurvedMirror
        ):
            warnings.warn(
                "For this method to work the first and last surfaces should be mirrors, using regular solver instead"
            )
            central_line_initial_parameters, central_line_successfully_traced = self.find_central_line_solver()

        else:
            theta_default, phi_default = self.default_initial_angles
            if self.t_is_trivial and self.p_is_trivial:
                solution_angles = np.array([theta_default, phi_default])
                central_line_successfully_traced = True
            elif (
                not self.t_is_trivial and self.p_is_trivial
            ):  # This syntax is for the case when we know the perturbation is only in
                # one dimensions, and the function can be reduced to one dimension as well. however, I think that the choice
                # of the one dimensions on which to solve the problem will not be consistent for different orientations of the
                # cavity, and so it might cause problems in the future
                # has a non-consistent entry in the
                def f_reduced(theta):
                    z, y = self.f_roots_standing_wave(np.array([theta, phi_default]))
                    if np.isnan(z):
                        z = np.inf * np.sign(theta)
                    return z

                solution = optimize.root_scalar(
                    f_reduced,
                    x0=theta_default,
                    x1=theta_default + 1e-10,
                    xtol=1e-12,
                )
                solution_angles = np.array([solution.root, phi_default])
                central_line_successfully_traced = solution.converged
            elif self.t_is_trivial and not self.p_is_trivial:

                def f_reduced(phi):
                    z, y = self.f_roots_standing_wave(np.array([theta_default, phi]))
                    if np.isnan(y):
                        y = np.inf * np.sign(phi)
                    return y

                solution = optimize.root_scalar(
                    f_reduced,
                    x0=phi_default,
                    x1=phi_default + 1e-9,
                    xtol=1e-12,
                )  # x0=np.array([self.default_initial_angles[1]])
                solution_angles = np.array([theta_default, solution.root])
                central_line_successfully_traced = solution.converged
            else:
                solution = optimize.root(self.f_roots_standing_wave, x0=np.array([*self.default_initial_angles]))
                solution_angles = solution.x
                central_line_successfully_traced = solution.success
            solution_ray = Ray(self.physical_surfaces[0].origin, -unit_vector_of_angles(*solution_angles))  # minus sign
            # central_line_successfully_traced = solution.
            # on the angles because we are searching now the intersection between the ray and the surface behind it.

            # Retrive the parameterization of the ray with respect to the first surface:
            ray_origin_on_first_surface = self.physical_surfaces[0].find_intersection_with_ray(solution_ray)
            solution_parameterization = self.physical_surfaces[0].get_parameterization(ray_origin_on_first_surface)
            central_line_initial_parameters = np.array(
                [solution_parameterization[0], solution_angles[0], solution_parameterization[1], solution_angles[1]]
            )

        return central_line_initial_parameters, central_line_successfully_traced

    def set_central_line(self, **kwargs) -> Tuple[np.ndarray, bool]:
        if self.standing_wave:
            central_line_initial_parameters, central_line_successfully_traced = self.find_central_line_standing_wave()
        elif self.use_brute_force_for_central_line:
            central_line_initial_parameters, central_line_successfully_traced = self.find_central_line_brute_force(
                **kwargs
            )
        else:
            central_line_initial_parameters, central_line_successfully_traced = self.find_central_line_solver()

        if central_line_successfully_traced:
            self.central_line_successfully_traced = central_line_successfully_traced
            origin_solution = self.arms[0].surface_0.parameterization(
                central_line_initial_parameters[0], central_line_initial_parameters[2]
            )  # theta, phi
            k_vector_solution = unit_vector_of_angles(
                central_line_initial_parameters[1], central_line_initial_parameters[3]
            )  # theta, phi
            central_line = Ray(origin_solution, k_vector_solution)
            # This line is to save the central line in the ray history, so that it can be plotted later.
            central_line = self.trace_ray(central_line)
            for i, arm in enumerate(self.arms):
                arm.central_line = central_line[i]
            self.central_line_successfully_traced = central_line_successfully_traced
        else:
            self.central_line_successfully_traced = False
            return central_line_initial_parameters, self.central_line_successfully_traced

    def set_mode_parameters(
        self,
        mode_parameters_first_arm: Optional[ModeParameters] = None,
        local_mode_parameters_first_surface: Optional[LocalModeParameters] = None,
    ):
        # Sets the mode parameters sequentially in all arms of the cavity. tries to find a mode solution for the cavity,
        # if it fails, it will set the resonating_mode_successfully_traced to False, and will use the input
        # local_mode_parameters_first_surface instead.
        if self.central_line_successfully_traced is None:
            self.set_central_line()
        if self.central_line_successfully_traced is False:
            self.resonating_mode_successfully_traced = False
            return None

        local_mode_parameters_current = local_mode_parameters_of_round_trip_ABCD(
            round_trip_ABCD=self.ABCD_round_trip, lambda_0_laser=self.lambda_0_laser, n=self.arms[0].n
        )
        if (
            local_mode_parameters_current.z_R[0] == 0 or local_mode_parameters_current.z_R[1] == 0
        ):  # When there is no solution,
            # the z_R value comes out as zero.
            self.resonating_mode_successfully_traced = False
            if local_mode_parameters_first_surface is not None or mode_parameters_first_arm is not None:  # if there is
                # no wave solution, but the user gave an input wave to the cavity, then just propagate it throughout
                # the cavity, even though it is not a wave solution.
                if mode_parameters_first_arm is not None:
                    # If the user preferred to give ModeParameters instead of LocalModeParameters, then convert it to
                    # LocalModeParameters.
                    local_mode_parameters_first_surface = mode_parameters_first_arm.local_mode_parameters(
                        (self.arms[0].surface_0.center - mode_parameters_first_arm.center[0])
                        @ mode_parameters_first_arm.k_vector
                    )
                local_mode_parameters_current = local_mode_parameters_first_surface

        # If there is a valid mode to start propagating, then propagate it through the cavity:
        if local_mode_parameters_current.z_R[0] != 0 and local_mode_parameters_current.z_R[1] != 0:
            for arm in self.arms:
                arm.mode_parameters_on_surface_0 = local_mode_parameters_current
                local_mode_parameters_current = arm.propagate_local_mode_parameters()
                arm.mode_principle_axes = self.principle_axes(arm.central_line.k_vector)
            self.resonating_mode_successfully_traced = True

    def principle_axes(self, k_vector: np.ndarray):
        # Returns two vectors that are orthogonal to k_vector and each other, one lives in the central line plane,
        # the other is perpendicular to the central line plane.
        if self.central_line_successfully_traced is None:
            self.set_central_line()
        # ATTENTION! THIS ASSUMES THAT ALL THE CENTRAL LINE arms ARE IN THE SAME PLANE.
        # I find the biggest psuedo z because if the first two k_vector are parallel, the cross product is zero and the
        # result of the cross product will be determined by arbitrary numerical errors.
        possible_pseudo_zs = [
            np.cross(self.central_line[0].k_vector, self.central_line[i].k_vector)
            for i in range(1, len(self.central_line))
        ]  # Points to the positive
        biggest_psuedo_z = possible_pseudo_zs[np.argmax([np.linalg.norm(pseudo_z) for pseudo_z in possible_pseudo_zs])]
        # biggest_psuedo_z = np.cross(self.central_line[0].k_vector, self.central_line[1].k_vector)
        if np.linalg.norm(biggest_psuedo_z) < 1e-14:
            pseudo_z = np.array([0, 0, 1])
        else:
            pseudo_z = normalize_vector(biggest_psuedo_z)
        pseudo_x = np.cross(pseudo_z, k_vector)
        principle_axes = np.stack([pseudo_z, pseudo_x], axis=-1).T  # [z_x, z_y, z_z], [x_x, x_y, x_z]
        return principle_axes

    def ray_of_initial_parameters(self, initial_parameters: np.ndarray):
        # Assumes initial_parameters is of the shape [..., 4] where the last axis of size for represents theta, theta,
        # (two numbers to represent the location and angle on the first surface) and theta, phi (two angles of the k_vector).
        k_vector_i = unit_vector_of_angles(theta=initial_parameters[..., 1], phi=initial_parameters[..., 3])
        origin_i = self.arms[0].surface_0.parameterization(t=initial_parameters[..., 0], p=initial_parameters[..., 2])
        input_ray = Ray(origin=origin_i, k_vector=k_vector_i)
        return input_ray  # input_ray.origin and input_ray.k_vector are of shape [..., 3] where the ... is the same as
        # the first axis of initial_parameters.

    def generate_spot_size_lines(self, dim=2, plane="xy"):
        if self.arms[0].mode_parameters is None:
            self.set_mode_parameters()
        list_of_spot_size_lines = []
        if self.resonating_mode_successfully_traced is True:
            for arm in self.arms:
                spot_size_lines_separated = generate_spot_size_lines(
                    arm.mode_parameters,
                    first_point=arm.central_line.origin,
                    last_point=arm.central_line.origin + arm.central_line.k_vector * arm.central_line.length,
                    principle_axes=arm.mode_principle_axes,
                    dim=dim,
                    plane=plane,
                )
                list_of_spot_size_lines.extend(spot_size_lines_separated)
        return list_of_spot_size_lines

    def set_initial_surface(self) -> Optional[Surface]:
        # adds a virtual surface on the first arm that is perpendicular to the beam and centered between the first two
        # physical_surfaces.
        if not isinstance(self.arms[0].surface_0, PhysicalSurface):
            return self.arms[0].surface_0
        # gets a surface that sits between the first two physical_surfaces, centered and perpendicular to the central line.
        if self.central_line is None:
            final_position_and_angles, success = self.set_central_line()
            if not success:
                # warnings.warn("Could not find central line, so no initial surface could be set.")
                return None
        middle_point = (self.central_line[0].origin + self.central_line[1].origin) / 2
        initial_surface = FlatSurface(outwards_normal=-self.central_line[0].k_vector, center=middle_point)

        first_leg = self.arms[0]
        first_leg_first_sub_leg = Arm(first_leg.surface_0, initial_surface)
        first_leg_second_sub_leg = Arm(initial_surface, first_leg.surface_1)
        if self.standing_wave:
            last_leg = self.arms[-1]
            last_leg_first_sub_leg = Arm(last_leg.surface_0, initial_surface)
            last_leg_second_sub_leg = Arm(initial_surface, last_leg.surface_1)
            legs_list = (
                [first_leg_second_sub_leg]
                + self.arms[1:-1]
                + [
                    last_leg_first_sub_leg,
                    last_leg_second_sub_leg,
                    first_leg_first_sub_leg,
                ]
            )
        else:
            legs_list = [first_leg_second_sub_leg] + self.arms[1:] + [first_leg_first_sub_leg]
        self.arms = legs_list
        # Now, after you found the initial_surface, we can retrace the central line, but now let it out from the
        # initial surface, instead of the first mirror.
        self.set_central_line(override_existing=True)
        return initial_surface

    def ABCD_round_trip_matrix_numeric(
        self, central_line_initial_parameters: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if central_line_initial_parameters is None:
            central_line_initial_parameters, success = self.set_central_line()
            if not success:
                raise ValueError("Could not find central line")

        if isinstance(self.arms[0].surface_0, PhysicalSurface):
            self.set_initial_surface()

        dr = 1e-9

        # The i'th, j'th element of optimize.approx_fprime is the derivative of the i'th component of the output with
        # respect to the j'th component of the input, which is exactly the definition of the i'th j'th element of the
        # ABCD matrix.

        def trace_ray_parametric_parameters_only(parameters_initial):
            parameters_final, _ = self.trace_ray_parametric(parameters_initial)
            return parameters_final

        ABCD_matrix = optimize.approx_fprime(central_line_initial_parameters, trace_ray_parametric_parameters_only, dr)
        return ABCD_matrix

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        axis_span: Optional[Union[float, np.ndarray]] = None,
        camera_center: Union[float, int] = -1,
        dim: int = 2,
        laser_color: str = "r",
        plane: str = "xy",
        plot_mode_lines: bool = True,
        plot_central_line: bool = True,
        additional_rays: Optional[List[Ray]] = None,
    ) -> plt.Axes:

        if axis_span is None:

            axes_range = np.array(
                [
                    np.max([m.center[0] for m in self.physical_surfaces])
                    - np.min([m.center[0] for m in self.physical_surfaces]),
                    np.max([m.center[1] for m in self.physical_surfaces])
                    - np.min([m.center[1] for m in self.physical_surfaces]),
                    np.max([m.center[2] for m in self.physical_surfaces])
                    - np.min([m.center[2] for m in self.physical_surfaces]),
                ]
            )

            if self.t_is_trivial and self.p_is_trivial and dim == 2:
                if (
                    self.arms[0].mode_parameters is not None
                    and np.min(self.arms[0].mode_parameters_on_surface_0.z_R) > 0
                ):
                    maximal_spot_size = np.max([arm.mode_parameters_on_surface_0.spot_size[0] for arm in self.arms])
                    axis_span = np.array([axes_range[0], 6 * maximal_spot_size])
                else:
                    axis_span = np.array([axes_range[0], 0.01])
            else:
                axes_range[axes_range == 0] = 1e-5
                axis_span = axes_range
        else:
            axis_span = np.array([axis_span, axis_span, axis_span])

        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            if dim == 3:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.add_subplot(111)

        if camera_center == -1:
            origin_camera = np.array(
                [
                    (
                        np.max([m.center[0] for m in self.physical_surfaces])
                        + np.min([m.center[0] for m in self.physical_surfaces])
                    )
                    / 2,
                    (
                        np.max([m.center[1] for m in self.physical_surfaces])
                        + np.min([m.center[1] for m in self.physical_surfaces])
                    )
                    / 2,
                    (
                        np.max([m.center[2] for m in self.physical_surfaces])
                        + np.min([m.center[2] for m in self.physical_surfaces])
                    )
                    / 2,
                ]
            )

        else:
            camera_center_int = int(np.floor(camera_center))
            if np.mod(camera_center, 1) == 0.5:
                origin_camera = (
                    self.arms[camera_center_int].surface_0.center + self.arms[camera_center_int].surface_1.center
                ) / 2
            else:
                origin_camera = self.surfaces[camera_center_int].center

        x_index, y_index = plane_name_to_xy_indices(plane)
        ax.set_xlim(
            origin_camera[x_index] - axis_span[0] * 0.55,
            origin_camera[x_index] + axis_span[0] * 0.55,
        )
        ax.set_ylim(
            origin_camera[y_index] - axis_span[1] * 0.55,
            origin_camera[y_index] + axis_span[1] * 0.55,
        )

        if self.central_line is not None and plot_central_line:
            for ray in self.central_line:
                ray.plot(
                    ax=ax,
                    dim=dim,
                    color=laser_color,
                    plane=plane,
                    linestyle="--",
                    alpha=0.8,
                )

        if additional_rays is not None:
            for i, ray in enumerate(additional_rays):
                ray.plot(ax=ax, dim=dim, plane=plane, linestyle="--", alpha=0.8, label=i)

        for i, surface in enumerate(self.surfaces):
            # If there is not information on the spot size of the element, plot it with default length:
            if self.arms[0].mode_parameters is None or np.any(self.arms[0].mode_parameters.z_R == 0):
                surface.plot(ax=ax, dim=dim, plane=plane)
            else:
                # If there is information on the spot size of the element, plot it with the spot size length*2.5:
                spot_size = self.arms[i].mode_parameters_on_surface_0.spot_size
                if plane == "xy":
                    spot_size = spot_size[1]
                else:
                    spot_size = spot_size[0]
                length = spot_size * 5
                surface.plot(ax=ax, dim=dim, plane=plane, length=length)

        if self.lambda_0_laser is not None and plot_mode_lines and self.arms[0].central_line is not None:
            try:
                spot_size_lines = self.generate_spot_size_lines(dim=dim, plane=plane)
                for line in spot_size_lines:
                    if dim == 2:
                        ax.plot(
                            line[0, :],
                            line[1, :],
                            color=laser_color,
                            linestyle="--",
                            alpha=0.8,
                            linewidth=0.5,
                        )
                    else:
                        ax.plot(
                            line[0, :],
                            line[1, :],
                            line[2, :],
                            color=laser_color,
                            linestyle="--",
                            alpha=0.8,
                            linewidth=0.5,
                        )
            except (FloatingPointError, AttributeError):
                # print("Mode was not successfully found, mode lines not plotted.")
                pass
        ax.grid()
        if additional_rays is not None:
            ax.legend()
        return ax

    # parameter_index: Union[Tuple[int, int], Tuple[List[int], List[int]]],
    #                    shift_value: Union[float, np.ndarray]
    def calculated_shifted_cavity_overlap_integral(
        self, perturbation_pointer: Union[PerturbationPointer, List[PerturbationPointer]]
    ) -> Tuple[np.ndarray]:
        # For a prturbation of more than one parameter, the first dimension of shift is the shift version, and the second dimension for the parameter index
        # For example, if shift = [[1e-6, 2e-6], [3e-6, 4e-6]], then the first perturbation is [1e-6, 2e-6] and the second is [3e-6, 4e-6].
        n_shifts = len(perturbation_pointer)
        overlaps = np.zeros(n_shifts, dtype=np.float64)
        for i in range(n_shifts):
            new_cavity = perturb_cavity(self, perturbation_pointer[i])
            try:
                overlap = calculate_cavities_overlap(cavity_1=self, cavity_2=new_cavity)
            except np.linalg.LinAlgError:
                continue
            overlaps[i] = np.abs(overlap)
        if n_shifts == 1:
            overlaps = overlaps[0]
        return overlaps

    def calculate_parameter_tolerance(
        self,
        perturbation_pointer: PerturbationPointer,
        initial_step: float = 1e-7,
        overlap_threshold: float = 0.9,
        accuracy: float = 1e-3,
    ) -> float:
        if np.isnan(self.arms[0].mode_parameters.NA[0]):
            warnings.warn("cavity has no mode even before perturbation, returning nan.")
            return np.nan

        def f(shift):
            resulting_overlap = self.calculated_shifted_cavity_overlap_integral(perturbation_pointer(shift))
            return resulting_overlap

        tolerance = functions_first_crossing_both_directions(
            f=f,
            initial_step=initial_step,
            crossing_value=overlap_threshold,
            accuracy=accuracy,
        )
        return tolerance

    def generate_tolerance_matrix(
        self,
        initial_step: float = 1e-7,
        overlap_threshold: float = 0.9,
        accuracy: float = 1e-3,
        perturbable_params_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        if perturbable_params_names is None:
            perturbable_params_names = self.perturbable_params_names
        tolerance_matrix = np.zeros((len(self.params), len(perturbable_params_names)))
        for element_index in tqdm(
            range(len(self.params)), desc="Tolerance Matrix - element index: ", disable=self.debug_printing_level < 1
        ):
            for tolerance_matrix_index, param_name in (
                pbar := tqdm(enumerate(perturbable_params_names), disable=self.debug_printing_level < 1)
            ):
                pbar.set_description(f"Tolerance Matrix - parameter index:  {param_name}")
                tolerance_matrix[element_index, tolerance_matrix_index] = self.calculate_parameter_tolerance(
                    perturbation_pointer=PerturbationPointer(element_index, param_name),
                    initial_step=initial_step,
                    overlap_threshold=overlap_threshold,
                    accuracy=accuracy,
                )
        return tolerance_matrix

    def generate_overlap_series(
        self,
        shifts: Union[np.ndarray, float],  # Float is interpreted as linspace's limits,
        # np.ndarray means that the element_index'th parameter_index'th element of shifts is the linspace limits of
        # the element_index'th parameter_index'th parameter.
        shift_size: int = 50,
        perturbable_params_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        if perturbable_params_names is None:
            perturbable_params_names = self.perturbable_params_names
        overlaps = np.zeros((len(self.params), len(perturbable_params_names), shift_size))
        for element_index in tqdm(
            range(len(self.params)), desc="Overlap Series - element_index", disable=self.debug_printing_level < 1
        ):
            for j, parameter_name in tqdm(
                enumerate(perturbable_params_names),
                desc="Overlap Series - parameter_index",
                disable=self.debug_printing_level < 1,
            ):
                if isinstance(shifts, (float, int)):
                    shift_series = np.linspace(-shifts, shifts, shift_size)
                else:
                    if np.isnan(shifts[element_index, j]):
                        shift_series = np.linspace(-1e-10, 1e-10, shift_size)
                    else:
                        shift_series = np.linspace(
                            -shifts[element_index, j],
                            shifts[element_index, j],
                            shift_size,
                        )
                # The condition inside is for the case it is a mirror and the parameter is n, and then we don't want
                # to draw it.
                if SurfacesTypes.has_refractive_index(self.params[element_index].surface_type) or parameter_name != ParamsNames.n_inside_or_after:
                    overlaps[element_index, j, :] = self.calculated_shifted_cavity_overlap_integral(
                        perturbation_pointer=PerturbationPointer(
                            element_index=element_index, parameter_name=parameter_name, perturbation_value=shift_series
                        )
                    )
        return overlaps

    def generate_overlaps_graphs(
        self,
        initial_step: float = 1e-6,
        overlap_threshold: float = 0.9,
        accuracy: float = 1e-3,
        arm_index_for_NA: int = 0,
        tolerance_matrix: Optional[np.ndarray] = None,
        overlaps_series: Optional[np.ndarray] = None,
        names: Optional[List[str]] = None,
        ax: Optional[np.ndarray] = None,
        perturbable_params_names: Optional[List[str]] = None,
    ):
        if names is None:
            names = self.names

        if perturbable_params_names is None:
            perturbable_params_names = self.perturbable_params_names

        if ax is None:
            fig, ax = plt.subplots(
                len(self.params),
                len(perturbable_params_names),
                figsize=(len(perturbable_params_names) * 5, len(self.params) * 2.1),
            )
        else:
            fig = ax.flatten()[0].get_figure()

        if tolerance_matrix is None:
            tolerance_matrix = self.generate_tolerance_matrix(
                initial_step=initial_step,
                overlap_threshold=overlap_threshold,
                accuracy=accuracy,
            )

        if overlaps_series is None:
            overlaps_series = self.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix), shift_size=30)
        plt.suptitle(f"NA={self.arms[arm_index_for_NA].mode_parameters.NA[0]:.3e}")

        for i in range(len(self.params)):
            for j, parameter_name in enumerate(perturbable_params_names):
                # The condition inside is for the case it is a mirror and the parameter is n, and then we don't want
                # to draw it.
                if parameter_name == ParamsNames.n_inside_or_after and np.isnan(tolerance_matrix[i, j]):
                    continue
                tolerance = tolerance_matrix[i, j]
                if tolerance == 0 or np.isnan(tolerance):
                    tolerance = initial_step
                tolerance_abs = np.abs(tolerance)
                shifts = np.linspace(-2 * tolerance_abs, 2 * tolerance_abs, overlaps_series.shape[2])

                ax[i, j].plot(shifts, overlaps_series[i, j, :])

                title = f"{names[i]}, {parameter_name}, tolerance: {tolerance_abs:.2e}"
                ax[i, j].set_title(title)
                if i == len(self.params) - 1:
                    ax[i, j].set_xlabel("Shift")
                if j == 0:
                    ax[i, j].set_ylabel("Overlap")
                ax[i, j].axvline(tolerance, color="g", linestyle="--")
                ax[i, j].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                ax[i, j].axhline(
                    overlap_threshold,
                    color="r",
                    linestyle="--",
                    linewidth=0.5,
                    alpha=0.5,
                )
                try:
                    min_value = np.nanmin(overlaps_series[i, j, :])
                    ax[i, j].set_ylim(1.1 * min_value - 0.1, 1.1 - 0.1 * min_value)
                except ValueError:
                    pass
        fig.tight_layout()
        return ax

    def thermal_transformation(self, **kwargs) -> Cavity:
        unheated_surfaces = []
        for i, surface in enumerate(self.physical_surfaces):
            unheated_surface = surface.thermal_transformation(
                P_laser_power=-self.power,
                w_spot_size=self.arms[i].mode_parameters_on_surface_0.spot_size[0],
                **kwargs,
            )
            unheated_surfaces.append(unheated_surface)

        # After heating the lens is not necessarily symmetrical, and so we have to decompose it to two surfaces.
        if self.names[0] is None:
            names = None
        else:
            names = copy.copy(self.names)
            for i, surface_type in [p.surface_type for p in self.params]:
                if surface_type == SurfacesTypes.thick_lens:
                    names.insert(i + 1, names[i] + "_2")
                    names[i] = names[i] + "_1"

        unheated_cavity = Cavity(
            physical_surfaces=unheated_surfaces,
            standing_wave=self.standing_wave,
            lambda_0_laser=self.lambda_0_laser,
            names=names,
            set_central_line=True,
            set_mode_parameters=True,
            set_initial_surface=False,
            t_is_trivial=self.t_is_trivial,
            p_is_trivial=self.p_is_trivial,
            power=0,
        )

        return unheated_cavity

    def analyze_thermal_transformation(self, arm_index_for_NA: int):
        N = 5
        boolean_array = np.eye(N).astype(bool)
        boolean_array = np.vstack((np.zeros((1, N), dtype=bool), np.ones((1, N), dtype=bool), boolean_array))
        cavities = []  # [self]
        NA_orgiginal = self.arms[arm_index_for_NA].mode_parameters.NA[0]
        NAs = np.zeros(N + 2)
        # NAs[0] = NA_orgiginal
        for i in range(N + 2):
            (
                curvature_transform_lens,
                n_surface_transform_lens,
                n_volumetric_transform_lens,
                z_transform_lens,
                transform_mirror,
            ) = boolean_array[i, :]
            unheated_cavity = self.thermal_transformation(
                curvature_transform_lens=curvature_transform_lens,
                n_surface_transform_lens=n_surface_transform_lens,
                n_volumetric_transform_lens=n_volumetric_transform_lens,
                z_transform_lens=z_transform_lens,
                transform_mirror=transform_mirror,
            )
            cavities.append(unheated_cavity)
            NAs[i] = unheated_cavity.arms[arm_index_for_NA].mode_parameters.NA[0]
        names_list = [
            "No transformation",
            "All Transformations",
            "Only lens curvature ",
            "Only lens n surface ",
            "Only lens n volumetric ",
            "Only lens z ",
            "Only mirror",
        ]
        results_dict = dict(zip(names_list, NA_orgiginal / NAs))
        return results_dict, cavities

    def specs(
        self,
        save_specs_name: Optional[str] = None,
        tolerance_matrix: Union[np.ndarray, bool] = False,
        print_specs: bool = False,
        contracted: bool = True,
    ):
        elements_array = self.to_array.T.copy()
        elements_array = np.real(elements_array) + np.pi * np.imag(elements_array)
        df_elements = pd.DataFrame(
            elements_array,
            columns=self.names,
            index=list(PRETTY_INDICES_NAMES.values()),
        )

        df_elements_stacked = stack_df_for_print(df_elements)
        NAs_list = []
        lengths_list = []
        df_arms_list = []
        for i, arm in enumerate(self.arms):
            if (
                self.standing_wave and i >= len(self.arms) // 2
            ):  # If it is a standing wave cavity, print only half of the arms, as the second half is the same arms in reverse
                break
            NAs_list.append(arm.mode_parameters.NA[0])
            lengths_list.append(arm.central_line.length)
            df_arms_list.append(arm.specs())

        df_cavity = pd.DataFrame(
            {
                "Parameter": [
                    "Id",
                    "Finesse",
                    "Free spectral range",
                    "Roundtrip power losses",
                    "Power decay rate",  # 'Amplitude amplification factor'
                ]
                + [f"NA_{i}" for i in range(len(NAs_list))]
                + [f"length_{i}" for i in range(len(lengths_list))],
                "Value": [
                    self.id,
                    self.finesse,
                    self.free_spectral_range,
                    self.roundtrip_power_losses,
                    self.power_decay_rate,
                ]
                + NAs_list
                + lengths_list,
            }
        )
        df_cavity["Element"] = "Cavity"
        # df_cavity['Category'] = 'Cavity'
        # If it is a standing wave cavity, print only half of the arms, as the second half is the same arms in reverse
        df_arms = pd.concat(df_arms_list)

        if isinstance(tolerance_matrix, bool):
            tolerance_matrix = np.abs(self.generate_tolerance_matrix())
        if self.p_is_trivial and self.t_is_trivial:
            index = [
                "Tolerance - axial displacement",
                "Tolerance - transversal displacement",
                "Tolerance - tilt angle",
                "Tolerance - radius of Curvature",
                "Tolerance - refractive Index",
            ]
        else:
            index = [PRETTY_INDICES_NAMES[param_name] for param_name in self.perturbable_params_names]
        df_tolerance = pd.DataFrame(tolerance_matrix.T, columns=self.names, index=index)
        df_tolerance_stacked = stack_df_for_print(df_tolerance)

        whole_df = pd.concat([df_elements_stacked, df_cavity, df_arms, df_tolerance_stacked])
        whole_df["Value"] = whole_df["Value"].apply(lambda x: signif(x, 6))
        whole_df.drop_duplicates(inplace=True)

        if contracted:
            whole_df = whole_df[
                ~whole_df["Parameter"].isin(
                    [
                        "Power decay rate",
                        "Roundtrip power losses",
                        "Free spectral range",
                        "Azimuthal angle [rads]",
                        "Curvature sign",
                        "Elevation angle [rads]",
                        "Poisson ratio",
                        "Surface type",
                        "x [m]",
                        "y [m]",
                        "z [m]",
                        "Angle of incidence_inside [deg]",
                    ]
                )
            ]
            whole_df = whole_df[whole_df["Value"] != 0]

        index = pd.MultiIndex.from_arrays([whole_df["Element"], whole_df["Parameter"]])
        whole_df.set_index(index, inplace=True)
        whole_df.drop(columns=["Parameter", "Element"], inplace=True)
        whole_df.sort_index(inplace=True)

        if save_specs_name is not None:
            whole_df.to_csv(
                f'data//cavities-specs//specs_{datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")}_{save_specs_name}.csv'
            )

        if print_specs:
            print(whole_df, end="\n\n")

        return whole_df


def generate_tolerance_of_NA(
    params_array: np.ndarray,
    parameter_index_for_NA_control: Tuple[int, int],
    arm_index_for_NA: int,
    parameter_values: np.ndarray,
    initial_step: float = 1e-6,
    overlap_threshold: float = 0.9,
    accuracy: float = 1e-3,
    lambda_0_laser: float = 1064e-9,
    standing_wave: bool = True,
    t_is_trivial: bool = False,
    p_is_trivial: bool = True,
    return_cavities: bool = False,
    debug_printing_level: int = 0,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[Cavity]]]:
    tolerance_matrix = np.zeros(
        (
            params_array.shape[0],
            params_to_perturbable_params_names(params_array, t_is_trivial and p_is_trivial),
            parameter_values.shape[0],
        )
    )
    NAs = np.zeros(parameter_values.shape[0])
    cavities = []
    for k, parameter_value in tqdm(
        enumerate(parameter_values), desc="tolerance_of_NA: parameter_value", disable=debug_printing_level < 1
    ):
        params_temp = params_array.copy()
        params_temp[parameter_index_for_NA_control] = parameter_value
        cavity = Cavity.from_params(
            params=params_temp,
            set_mode_parameters=True,
            lambda_0_laser=lambda_0_laser,
            standing_wave=standing_wave,
            t_is_trivial=t_is_trivial,
            p_is_trivial=p_is_trivial,
            debug_printing_level=debug_printing_level,
        )
        if np.any(np.isnan(cavity.mode_parameters[arm_index_for_NA].NA)) or np.any(
            cavity.mode_parameters[arm_index_for_NA].NA == 0
        ):
            continue
        NAs[k] = cavity.mode_parameters[arm_index_for_NA].NA[0]  # ARBITRARY
        cavities.append(cavity)
        tolerance_matrix[:, :, k] = cavity.generate_tolerance_matrix(
            initial_step=initial_step,
            overlap_threshold=overlap_threshold,
            accuracy=accuracy,
        )
    if return_cavities:
        return NAs, tolerance_matrix, cavities
    else:
        return NAs, tolerance_matrix


def plot_tolerance_of_NA(
    params: Optional[np.ndarray] = None,
    parameter_index_for_NA_control: Optional[Tuple[int, int]] = None,
    arm_index_for_NA: Optional[int] = None,
    parameter_values: Optional[np.ndarray] = None,
    initial_step: Optional[float] = 1e-6,
    overlap_threshold: Optional[float] = 0.9,
    accuracy: Optional[float] = 1e-3,
    names: Optional[List[str]] = None,
    lambda_0_laser: Optional[float] = 1064e-9,
    standing_wave: Optional[bool] = True,
    t_is_trivial: bool = False,
    p_is_trivial: bool = True,
    NAs: Optional[np.ndarray] = None,
    tolerance_matrix: Optional[np.ndarray] = None,
):
    if tolerance_matrix is None:
        NAs, tolerance_matrix = generate_tolerance_of_NA(
            params_array=params,
            parameter_index_for_NA_control=parameter_index_for_NA_control,
            arm_index_for_NA=arm_index_for_NA,
            parameter_values=parameter_values,
            initial_step=initial_step,
            overlap_threshold=overlap_threshold,
            accuracy=accuracy,
            lambda_0_laser=lambda_0_laser,
            standing_wave=standing_wave,
        )
    tolerance_matrix = np.abs(tolerance_matrix)
    number_of_params = len(params_to_perturbable_params_names(params, t_is_trivial and p_is_trivial))
    fig, ax = plt.subplots(
        tolerance_matrix.shape[0],
        number_of_params,
        figsize=(number_of_params * 5, tolerance_matrix.shape[0] * 2),
    )
    if names is None:
        names = [None for _ in range(params.shape[0])]
    for i in range(tolerance_matrix.shape[0]):
        for j in range(number_of_params):
            ax[i, j].plot(NAs, tolerance_matrix[i, j, :], color="g")
            title = f"{names[i]}, {INDICES_DICT_INVERSE[j]}"
            ax[i, j].set_title(title)
            if i == tolerance_matrix.shape[0] - 1:
                ax[i, j].set_xlabel("NA")
            if j == 0:
                ax[i, j].set_ylabel("Tolerance")
            ax[i, j].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax[i, j].set_yscale("log")
    fig.tight_layout()
    return fig, ax


def plot_tolerance_of_NA_same_plot(
    params: Optional[np.ndarray] = None,
    parameter_index_for_NA_control: Optional[Tuple[int, int]] = None,
    arm_index_for_NA: Optional[int] = None,
    parameter_values: Optional[np.ndarray] = None,
    initial_step: Optional[float] = 1e-6,
    overlap_threshold: Optional[float] = 0.9,
    accuracy: Optional[float] = 1e-3,
    names: Optional[List[str]] = None,
    lambda_0_laser: Optional[float] = 1064e-9,
    standing_wave: Optional[bool] = True,
    NAs: Optional[np.ndarray] = None,
    tolerance_matrix: Optional[np.ndarray] = None,
    ax: plt.Axes = None,
    t_and_p_are_trivial: bool = False,
):
    if tolerance_matrix is None:
        NAs, tolerance_matrix = generate_tolerance_of_NA(
            params_array=params,
            parameter_index_for_NA_control=parameter_index_for_NA_control,
            arm_index_for_NA=arm_index_for_NA,
            parameter_values=parameter_values,
            initial_step=initial_step,
            overlap_threshold=overlap_threshold,
            accuracy=accuracy,
            lambda_0_laser=lambda_0_laser,
            standing_wave=standing_wave,
        )
    tolerance_matrix = np.abs(tolerance_matrix)
    n_elements = tolerance_matrix.shape[0]
    if ax is None:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    if names is None:
        names = [None for _ in range(n_elements)]

    j_ranges = [
        [INDICES_DICT["x"]],
        [INDICES_DICT["y"], INDICES_DICT["z"]],
        [INDICES_DICT["theta"], INDICES_DICT["phi"]],
        [INDICES_DICT["r_1"], INDICES_DICT["n_inside_or_after"]],
    ]
    titles = [
        "Axial Position",
        "Transverse Position",
        "Tilt Angles",
        "Radius and Index",
    ]

    if t_and_p_are_trivial:
        j_ranges[1].remove(INDICES_DICT["z"])
        j_ranges[2].remove(INDICES_DICT["theta"])

    for l, a in enumerate(ax.ravel()):
        for i in range(n_elements):
            for j in j_ranges[l]:
                # The condition inside is for the case it is a mirror and the parameter is n, and then we don't want
                # to draw it.
                if not (
                    j == INDICES_DICT["n_inside_or_after"]
                    and (np.isnan(tolerance_matrix[i, j, 0]) or tolerance_matrix[i, j, 0] == 0)
                ):
                    linewidth = 1 + 0.2 * (n_elements - i - 1)
                    print(linewidth)
                    a.plot(
                        NAs,
                        tolerance_matrix[i, j, :],
                        linewidth=linewidth,
                        label=f"{names[i]}, {INDICES_DICT_INVERSE[j]}",
                    )

        a.set_xlabel("NA")
        a.set_ylabel("Tolerance")
        # a.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        a.set_yscale("log")
        a.set_xscale("log")
        a.grid(True)
        a.set_title(titles[l])
        a.legend()
    return ax


def calculate_gaussian_parameters_on_surface(surface: FlatSurface, mode_parameters: ModeParameters):
    # Derivations to all this mathematical s**theta is the LabArchives: https://mynotebook.labarchives.com/MTM3NjE3My41fDEwNTg1OTUvMTA1ODU5NS9Ob3RlYm9vay8zMjQzMzA0MzY1fDM0OTMzNjMuNQ==/page/11290221-13
    intersection_point = surface.find_intersection_with_ray(mode_parameters.ray)
    intersection_point = intersection_point[0, :]
    z_minus_z_0 = np.linalg.norm(intersection_point - mode_parameters.center, axis=1)
    q_u, q_v = z_minus_z_0 + 1j * mode_parameters.z_R
    k = 2 * np.pi / mode_parameters.lambda_0_laser

    # Those are the vectors that define the mode and the surface: r_0 is the surface's center with respect to the mode's
    # center, t_hat and p_hat are the unit vectors that span the surface, k_hat is the mode's k vector,
    # u_hat and v_hat are the principle axes of the mode.
    r_0 = surface.center - mode_parameters.center[0, :]  # Technically there are two centers, but their difference is
    # only in the k_hat direction, which doesn'theta make a difference on the projection on the two principle axes of the
    # mode, and for the projection of the k_hat vector we anyway need to set an arbitrary 0, so we can just take the
    # first center.
    t_hat, p_hat = normalize_vector(surface.parameterization(1, 0) - surface.parameterization(0, 0)), normalize_vector(
        surface.parameterization(0, 1) - surface.parameterization(0, 0)
    )
    k_hat = mode_parameters.k_vector
    u_hat_v_hat = mode_parameters.principle_axes
    u_hat = u_hat_v_hat[0, :]
    v_hat = u_hat_v_hat[1, :]

    # The mode as a function of the surface's parameterization:
    # exp([theta,phi] @ A_2 @ [theta,phi] + b @ [theta,phi] + c

    A = (
        1j
        * k
        * np.array(
            [
                [
                    (t_hat @ u_hat) ** 2 / q_u + (t_hat @ v_hat) ** 2 / q_v,
                    (t_hat @ u_hat) * (p_hat @ u_hat) / q_u + (t_hat @ v_hat) * (p_hat @ v_hat) / q_v,
                ],
                [
                    (t_hat @ u_hat) * (p_hat @ u_hat) / q_u + (t_hat @ v_hat) * (p_hat @ v_hat) / q_v,
                    (p_hat @ u_hat) ** 2 / q_u + (p_hat @ v_hat) ** 2 / q_v,
                ],
            ]
        )
    )
    b = (
        -(1 / 2)
        * 1j
        * k
        * np.array(
            [
                (k_hat @ t_hat) + 2 * (r_0 @ u_hat) * (t_hat @ u_hat) / q_u + 2 * (r_0 @ v_hat) * (t_hat @ v_hat) / q_v,
                (k_hat @ p_hat) + 2 * (r_0 @ u_hat) * (p_hat @ u_hat) / q_u + 2 * (r_0 @ v_hat) * (p_hat @ v_hat) / q_v,
            ]
        )
    )
    c = -(1 / 2) * 1j * k * ((k_hat @ r_0) + (r_0 @ u_hat) ** 2 / q_u + (r_0 @ v_hat) ** 2 / q_v)

    return A, b, c


def evaluate_cavities_modes_on_surface(cavity_1: Cavity, cavity_2: Cavity):
    # Chooses a plane on which to evaluate the modes, and calculate the gaussian coefficients of the modes on that plane
    # for both cavities.
    correct_modes = True
    for cavity in [cavity_1, cavity_2]:
        if cavity.arms[0].mode_parameters is None:
            try:
                cavity.set_mode_parameters()
            except FloatingPointError:
                correct_modes = False
                break

    mode_parameters_1 = cavity_1.arms[0].mode_parameters
    mode_parameters_2 = cavity_2.arms[0].mode_parameters

    if mode_parameters_1 is None or mode_parameters_2 is None:
        A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes = 0, 0, 0, 0, 0, 0, 0, False
        return A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes

    NAs = np.concatenate((mode_parameters_1.NA, mode_parameters_2.NA))
    if (
        cavity_1.central_line_successfully_traced is False
        or cavity_2.central_line_successfully_traced is False
        or correct_modes is False
        or np.any(np.isnan(NAs))
    ):
        A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes = 0, 0, 0, 0, 0, 0, 0, False
        return A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes

    # Note that the waist migh be outside the arm, but even if it is, the mode is still valid.
    cavity_1_waist_pos = mode_parameters_1.center[0, :]  #  we take the waist of the first transversal direction
    P1 = FlatSurface(center=cavity_1_waist_pos, outwards_normal=mode_parameters_1.k_vector)
    try:
        A_1, b_1, c_1 = calculate_gaussian_parameters_on_surface(P1, mode_parameters_1)
        A_2, b_2, c_2 = calculate_gaussian_parameters_on_surface(P1, mode_parameters_2)
    except FloatingPointError:
        A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes = 0, 0, 0, 0, 0, 0, 0, False
    return A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes


def calculate_cavities_overlap(cavity_1: Cavity, cavity_2: Cavity) -> float:
    A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes = evaluate_cavities_modes_on_surface(cavity_1, cavity_2)
    if correct_modes is False:
        return np.nan
    else:
        return gaussians_overlap_integral(A_1, A_2, b_1, b_2, c_1, c_2)


def evaluate_gaussian(A: np.ndarray, b: np.ndarray, c: complex, axis_span: float, N: int = 100):
    x = np.linspace(-axis_span, axis_span, N)
    y = np.linspace(-axis_span, axis_span, N)
    X, Y = np.meshgrid(x, y)
    R = np.stack([X, Y], axis=2)
    # mu = np.array([x_2, y_2])
    # R_shifted = R - mu[None, None, :]
    R_normed_squared = np.einsum("ijk,kl,ijl->ij", R, A, R)
    functions_values = safe_exponent(-(1 / 2) * R_normed_squared + np.einsum("k,ijk->ij", b, R) + c)
    return functions_values


def perturb_cavity(
    cavity: Cavity,
    perturbation_pointer: Union[PerturbationPointer, List[PerturbationPointer]],
    **kwargs,  # For the initialization of the new cavity
):
    new_params = copy.deepcopy(cavity.params)
    for perturbation_pointer_temp in perturbation_pointer:
        current_value = getattr(
            new_params[perturbation_pointer_temp.element_index], perturbation_pointer_temp.parameter_name
        )
        new_value = current_value + perturbation_pointer_temp.perturbation_value
        # Set the new value back to the attribute
        setattr(
            new_params[perturbation_pointer_temp.element_index], perturbation_pointer_temp.parameter_name, new_value
        )

    parameters_names = [p.parameter_name for p in perturbation_pointer_temp]

    # If the original cavity was symmetrical in the theta axis or the phi axis, and the perturbation does not disturb this
    # symmetry, then the new cavity is also symmetrical in the theta axis or the phi axis:
    perturbance_in_z = [1 for i in parameters_names if i in [ParamsNames.z, ParamsNames.theta]]
    perturbance_in_y = [1 for i in parameters_names if i in [ParamsNames.y, ParamsNames.phi]]
    perturbance_in_z = bool(len(perturbance_in_z))
    perturbance_in_y = bool(len(perturbance_in_y))

    t_is_trivial = cavity.t_is_trivial and not perturbance_in_z
    p_is_trivial = cavity.p_is_trivial and not perturbance_in_y

    new_cavity = Cavity.from_params(
        params=new_params,
        standing_wave=cavity.standing_wave,
        lambda_0_laser=cavity.lambda_0_laser,
        t_is_trivial=t_is_trivial,
        p_is_trivial=p_is_trivial,
        use_brute_force_for_central_line=cavity.use_brute_force_for_central_line,
        debug_printing_level=cavity.debug_printing_level,
        use_paraxial_ray_tracing=cavity.use_paraxial_ray_tracing,
        **kwargs,
    )
    return new_cavity


def plot_gaussian_subplot(
    A: np.ndarray,
    b: np.ndarray,
    c: float,
    axis_span: float = 0.0005,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
):
    if ax is None:
        fig, ax = plt.subplots()
    functions_values = evaluate_gaussian(A, b, c, axis_span)
    im = ax.imshow(np.real(functions_values))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    return fig, ax


def plot_2_gaussians_subplots(
    A_1: np.ndarray,
    A_2: np.ndarray,
    # mu_1: np.ndarray, mu_2: np.ndarray, # Seems like I don't need the mus.
    b_1: np.ndarray,
    b_2: np.ndarray,
    c_1: float,
    c_2: float,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    axis_span: float = 0.0005,
    title: Optional[str] = "",
):
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    plot_gaussian_subplot(A_1, b_1, c_1, axis_span, fig, ax[0])
    plot_gaussian_subplot(A_2, b_2, c_2, axis_span, fig, ax[1])
    if title is not None:
        fig.suptitle(title)


def plot_2_gaussians_colors(
    A_1: np.ndarray,
    A_2: np.ndarray,
    b_1: np.ndarray,
    b_2: np.ndarray,
    c_1: float,
    c_2: float,
    ax: Optional[plt.Axes] = None,
    axis_span: float = 0.0005,
    title: Optional[str] = "",
    real_or_abs: str = "abs",
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    first_gaussian_values = evaluate_gaussian(A_1, b_1, c_1, axis_span)
    second_gaussian_values = evaluate_gaussian(A_2, b_2, c_2, axis_span)
    first_gaussian_values = first_gaussian_values / np.max(np.abs(first_gaussian_values))
    second_gaussian_values = second_gaussian_values / np.max(np.abs(second_gaussian_values))
    third_color_channel = np.zeros_like(first_gaussian_values)
    rgb_image = np.stack([first_gaussian_values, second_gaussian_values, third_color_channel], axis=2)
    if real_or_abs == "abs":
        rgb_image = np.clip(np.abs(rgb_image), 0, 1)
    else:
        rgb_image = np.real(rgb_image)
    ax.imshow(rgb_image, extent=[-axis_span, axis_span, -axis_span, axis_span])
    ax.set_title(title)


def plot_2_cavity_perturbation_overlap(
    cavity: Cavity,
    parameter_index: Optional[Tuple[int, int]] = None,
    shift_value: Optional[float] = None,
    second_cavity: Cavity = None,
    ax: Optional[plt.Axes] = None,
    axis_span: float = 0.0005,
):
    if second_cavity is None:
        second_cavity = perturb_cavity(cavity, parameter_index, shift_value)

    A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_mode = evaluate_cavities_modes_on_surface(cavity, second_cavity)
    if correct_mode:
        plot_2_gaussians_colors(
            A_1,
            A_2,
            b_1,
            b_2,
            c_1,
            c_2,
            ax=ax,
            axis_span=axis_span,
            title="Cavity perturbation overlap",
            real_or_abs="abs",
        )


def evaluate_gaussian_3d(points: np.ndarray, mode_parameters: ModeParameters):
    center = mode_parameters.center[0, :]
    points_relative = points - center
    u_vec = mode_parameters.principle_axes[0, :]
    v_vec = mode_parameters.principle_axes[1, :]
    k_vec = mode_parameters.k_vector
    k_projection = points_relative @ k_vec
    u_projection = points_relative @ u_vec
    v_projection = points_relative @ v_vec
    q = k_projection[:, :, None] + 1j * mode_parameters.z_R[None, None, :]
    q_u = q[:, :, 0]
    q_v = q[:, :, 1]
    k = 2 * np.pi / mode_parameters.lambda_0_laser
    integrand = -1j * k / 2 * (u_projection**2 / q_u + v_projection**2 / q_v + k_projection)
    gaussian = safe_exponent(integrand)
    return gaussian


def find_distance_to_first_crossing_positive_side(
    shifts: np.ndarray, overlaps: np.ndarray, crossing_value: float = 0.9
):
    overlaps_under_crossing = overlaps < crossing_value
    if np.any(overlaps_under_crossing):
        first_overlap_crossing = np.argmax(overlaps_under_crossing)
        if first_overlap_crossing == 0:
            crossing_shift = np.nan
        else:
            crossing_shift = interval_parameterization(
                shifts[first_overlap_crossing - 1],
                shifts[first_overlap_crossing],
                (crossing_value - overlaps[first_overlap_crossing - 1])
                / (overlaps[first_overlap_crossing] - overlaps[first_overlap_crossing - 1]),
            )
    else:
        crossing_shift = np.nan
    return crossing_shift


def find_distance_to_first_crossing(shifts: np.ndarray, overlaps: np.ndarray, crossing_value: float = 0.9):
    # Assumes shifts is ascending and that overlaps[i] is the overlap of shift[i]
    positive_shifts = shifts[shifts >= 0]
    negative_shifts = -shifts[shifts <= 0]
    positive_shifts_overlaps = overlaps[shifts >= 0]
    negative_shifts_overlaps = overlaps[shifts <= 0]
    negative_shifts_overlaps = negative_shifts_overlaps[::-1]
    negative_shifts = negative_shifts[::-1]
    crossing_positive_shift = find_distance_to_first_crossing_positive_side(
        positive_shifts, positive_shifts_overlaps, crossing_value=crossing_value
    )
    crossing_negative_shift = find_distance_to_first_crossing_positive_side(
        negative_shifts, negative_shifts_overlaps, crossing_value=crossing_value
    )
    if np.isnan(crossing_negative_shift):
        crossing_shift = crossing_positive_shift
    elif np.isnan(crossing_positive_shift):
        crossing_shift = -crossing_negative_shift
    elif crossing_negative_shift < crossing_positive_shift:
        crossing_shift = -crossing_negative_shift
    elif crossing_positive_shift <= crossing_negative_shift:
        crossing_shift = crossing_positive_shift
    else:
        raise ValueError("Debug me")
    return crossing_shift


def functions_first_crossing_both_directions(
    f: Callable,
    initial_step: float,
    crossing_value: float = 0.9,
    accuracy: float = 0.001,
) -> float:
    positive_step = functions_first_crossing(f, initial_step, crossing_value, accuracy)
    negative_step = functions_first_crossing(lambda x: f(-x), initial_step, crossing_value, accuracy)
    if positive_step < negative_step:
        return positive_step
    else:
        return -negative_step


def match_a_mirror_to_mode(
    mode: ModeParameters, z, thermal_properties: MaterialProperties
) -> Union[FlatMirror, CurvedMirror]:
    if z == 0:
        mirror = FlatMirror(
            center=mode.center[0, :],
            outwards_normal=mode.k_vector,
            thermal_properties=thermal_properties,
        )
    else:
        R_z_inverse = np.abs(z / (z**2 + mode.z_R[0] ** 2))
        center = mode.center[0, :] + mode.k_vector * z
        outwards_normal = mode.k_vector * np.sign(z)
        mirror = CurvedMirror(
            center=center,
            outwards_normal=outwards_normal,
            radius=R_z_inverse**-1,
            thermal_properties=thermal_properties,
        )
    return mirror


def local_mode_2_of_lens_parameters(
    lens_parameters: np.ndarray, local_mode_1: LocalModeParameters
):  # les_parameters = [r, n, w]
    R, w, n = lens_parameters
    params = np.array([0, 0, 0, 0, R, n, w, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    surface_0, surface_2 = generate_lens_from_params(params)
    ABCD_first = surface_0.ABCD_matrix(cos_theta_incoming=1)
    ABCD_between = ABCD_free_space(w)
    ABCD_second = surface_2.ABCD_matrix(cos_theta_incoming=1)
    ABCD_total = ABCD_second @ ABCD_between @ ABCD_first
    propagated_mode = propagate_local_mode_parameter_through_ABCD(local_mode_1, ABCD_total, n_1=1, n_2=1)
    return propagated_mode


def match_a_lens_parameters_to_modes(
    local_mode_1: LocalModeParameters,
    local_mode_2: LocalModeParameters,
    fixed_n_lens: Optional[float] = None,
    fix_z_2: bool = False,
):
    def f_roots(lens_parameters: np.ndarray):
        if fixed_n_lens is not None:
            lens_parameters = np.array([lens_parameters[0], lens_parameters[1], fixed_n_lens])
        propagated_mode = local_mode_2_of_lens_parameters(lens_parameters, local_mode_1)
        q_error = propagated_mode.q[0] - local_mode_2.q[0]
        if not fix_z_2:  # if we don't fix z_2, then the error in z_2 is set to 0, regardless of the actual value.
            q_error = 1j * np.imag(q_error)

        if fixed_n_lens is not None:
            return np.array([np.real(q_error), np.imag(q_error)])
        else:
            return np.array([np.real(q_error), np.imag(q_error), 0])

    if fixed_n_lens is not None:
        lens_parameters = optimize.fsolve(f_roots, np.array([1e-2, 1e-3]))
        lens_parameters = np.array([lens_parameters[0], lens_parameters[1], fixed_n_lens])
    else:
        lens_parameters = optimize.fsolve(f_roots, np.array([1e-2, 1e-3, 1.6]))

    resulted_mode_2_parameters = local_mode_2_of_lens_parameters(lens_parameters, local_mode_1)
    return lens_parameters, resulted_mode_2_parameters


def compare_2_cylindrical_cavities(
    params_1: np.ndarray,
    params_2: np.ndarray,
    generate_tolerance_of_NA_dict: dict = {},
    cavities_names: Tuple[str] = ("cavity 1", "cavity 2"),
    elements_names: Tuple[str] = ("Long Arm Mirror", "Lens", "Short Arm Mirror"),
):
    NAs_1, tolerance_matrix_1 = generate_tolerance_of_NA(
        params_1, t_is_trivial=True, p_is_trivial=True, **generate_tolerance_of_NA_dict
    )
    NAs_2, tolerance_matrix_2 = generate_tolerance_of_NA(
        params_2, t_is_trivial=True, p_is_trivial=True, **generate_tolerance_of_NA_dict
    )
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    plot_tolerance_of_NA_same_plot(
        params=params_1,
        names=[element_name + " " + cavities_names[0] for element_name in elements_names],
        NAs=NAs_1,
        tolerance_matrix=np.abs(tolerance_matrix_1),
        ax=ax,
        t_and_p_are_trivial=True,
    )
    plot_tolerance_of_NA_same_plot(
        params=params_2,
        names=[element_name + " " + cavities_names[1] for element_name in elements_names],
        NAs=NAs_1,
        tolerance_matrix=np.abs(tolerance_matrix_2),
        ax=ax,
        t_and_p_are_trivial=True,
    )
    return ax


def maximize_overlap(
    cavity: Cavity,
    perturbed_parameter_index: Tuple[int, int],
    perturbation_value: float,
    control_parameters_indices: Tuple[List[int], List[int]],
    print_progress: bool = False,
):
    perturbed_cavity = perturb_cavity(cavity, perturbed_parameter_index, perturbation_value)
    original_overlap = np.abs(calculate_cavities_overlap(cavity_1=cavity, cavity_2=perturbed_cavity))
    if print_progress:
        print("Original overlap:", original_overlap)
        I = 0

    def controlled_overlap(control_parameters_values: np.ndarray):
        corrected_cavity = perturb_cavity(
            perturbed_cavity, control_parameters_indices, control_parameters_values
        )  #  * 1e-3
        overlap = calculate_cavities_overlap(cavity_1=cavity, cavity_2=corrected_cavity)
        overlap_abs_minus = np.nan_to_num(-np.abs(overlap), nan=2)
        if print_progress:
            nonlocal I
            I += 1
            print(
                "Iteration",
                I,
                "control_parameters_values",
                control_parameters_values,
                "overlap:",
                np.abs(overlap),
            )
        return overlap_abs_minus

    best_overlap = optimize.minimize(controlled_overlap, x0=np.zeros(len(control_parameters_indices[0])), tol=1e-6)
    # best_overlap.x *= 1e-3
    if print_progress:
        print("Number of iterations:", I)
    # best_overlap = optimize.fsolve(controlled_overlap, x0=np.zeros(len(control_parameters_indices[0])))

    return best_overlap, original_overlap


def find_minimal_width_for_spot_size_and_radius(radius, spot_size_radius, T_edge=1e-3, h_divided_by_spot_size=2.8):
    # relies on the derivation in figures/lens thickness calculation.jpg
    h = h_divided_by_spot_size * spot_size_radius
    try:
        dT_c = radius * (1 - np.sqrt(1 - h**2 / radius**2))
        minimal_T_c = 2 * dT_c + T_edge
        return minimal_T_c
    except FloatingPointError:
        warnings.warn(
            "The spot size radius is too large for the given radius, returning nan",
        )
        return np.nan


def calculate_incidence_angle(surface: Surface, mode_parameters: ModeParameters) -> float:
    # Calculates the incidence angle betweer the beam at the E=E_0e^-1 lateral_position (one spot size away from it's optical axis)
    # and the surface of an optical element
    if isinstance(surface, FlatSurface):
        raise NotImplementedError("The function is not implemented for flat surfaces")

    surface_center_to_waist_position_vector = mode_parameters.center[0, :] - surface.center
    from_the_convex_side = np.sign(surface.outwards_normal @ surface_center_to_waist_position_vector)
    surface_to_waist_distance_signed = np.linalg.norm(surface_center_to_waist_position_vector) * from_the_convex_side

    angle_of_incidence = np.arcsin(
        ((surface.radius + surface_to_waist_distance_signed) * mode_parameters.NA[0]) / surface.radius
    )

    angle_of_incidence_deg = np.degrees(angle_of_incidence)
    return angle_of_incidence_deg


def generate_spot_size_lines(
    mode_parameters: ModeParameters,
    first_point: np.ndarray,
    last_point: np.ndarray,
    dim: int = 2,
    plane: str = "xy",
    principle_axes: Optional[np.ndarray] = None,
):
    if mode_parameters.principle_axes is not None and principle_axes is None:
        principle_axes = mode_parameters.principle_axes
    elif plane == "xy" and principle_axes is None:
        principle_axes = np.array([[0, 0, 1], [0, -1, 0]])
    central_line = Ray(
        origin=first_point,
        k_vector=mode_parameters.k_vector,
        length=np.linalg.norm(last_point - first_point),
    )
    t = np.linspace(0, central_line.length, 100)  # 100 is always enough
    ray_points = central_line.parameterization(t=t)
    z_minus_z_0 = np.linalg.norm(ray_points[:, np.newaxis, :] - mode_parameters.center, axis=2)  # Before
    # the norm the size is 100 | 2 | 3 and after it is 100 | 2 (100 points for in_plane and out_of_plane
    # dimensions)
    sign = np.array([1, -1])
    spot_size_value = spot_size(z_minus_z_0, mode_parameters.z_R, mode_parameters.lambda_0_laser, mode_parameters.n)
    spot_size_lines = (
        ray_points[:, np.newaxis, np.newaxis, :]
        + spot_size_value[:, :, np.newaxis, np.newaxis]
        * principle_axes[np.newaxis, :, np.newaxis, :]
        * sign[np.newaxis, np.newaxis, :, np.newaxis]
    )  # The size is 100 (n_points) | 2 (axis, []) | 2 (sign, [1, -1]) | 3 (coordinate, [x,y,z])
    if dim == 2:
        if plane in ["xy", "yx"]:
            relevant_axis_index = 1
            relevant_diminsions = [0, 1]
        elif plane in ["xz", "zx"]:
            relevant_axis_index = 0
            relevant_diminsions = [0, 2]
        else:
            relevant_axis_index = 0
            relevant_diminsions = [1, 2]
        spot_size_lines = spot_size_lines[:, relevant_axis_index, :, relevant_diminsions]  # Drop the z axis,
        # and drop the lines of the transverse axis the size is:
        # 2 (selected spatial axes) | 100 (n_points) | 2 (sign, [1, -1]
        spot_size_lines_separated = [spot_size_lines[:, :, 0], spot_size_lines[:, :, 1]]
    else:
        spot_size_lines_separated = [
            spot_size_lines[:, 0, 0, :],
            spot_size_lines[:, 0, 1, :],
            spot_size_lines[:, 1, 0, :],
            spot_size_lines[:, 1, 1, :],
        ]  # Each
        # element is a  100 | 3 array.

    return spot_size_lines_separated


def find_equal_angles_surface(
    mode_before_lens: ModeParameters,
    surface_0: CurvedRefractiveSurface,
    T_edge: float = 1e-3,
    h: float = 3.875e-3,
    lambda_0_laser: float = 1064e-9,
) -> CurvedRefractiveSurface:
    mode_parameters_just_before_surface_0 = mode_before_lens.local_mode_parameters(
        np.linalg.norm(surface_0.center - mode_before_lens.center[0])
    )
    first_angle_of_incidence = calculate_incidence_angle(
        surface=surface_0,
        mode_parameters=mode_before_lens,
    )
    dT_c_0 = dT_c_of_a_lens(R=surface_0.radius, h=h)
    mode_parameters_right_after_surface_0 = propagate_local_mode_parameter_through_ABCD(
        mode_parameters_just_before_surface_0,
        surface_0.ABCD_matrix(cos_theta_incoming=1),
        n_1=surface_0.n_1,
        n_2=surface_0.n_2,
    )

    def match_surface_to_radius(R_1: float) -> CurvedRefractiveSurface:
        T_c = dT_c_0 + T_edge + dT_c_of_a_lens(R=R_1, h=h)
        center_1 = surface_0.center + surface_0.inwards_normal * T_c
        second_surface = CurvedRefractiveSurface(
            radius=R_1,
            outwards_normal=-surface_0.outwards_normal,
            center=center_1,
            n_1=surface_0.n_2,
            n_2=1,
            curvature_sign=-1 * surface_0.curvature_sign,
            thickness=T_c / 2,
        )
        return second_surface

    def f_for_root(R_1: np.ndarray) -> float:  # ARBITRARY - ASSUMES CONVEX LENS
        # R_1 = R_1[0]
        second_surface = match_surface_to_radius(R_1)
        arm = Arm(
            surface_0=surface_0,
            surface_1=second_surface,
            central_line=Ray(
                origin=surface_0.center,
                k_vector=normalize_vector(second_surface.center - surface_0.center),
                length=np.linalg.norm(second_surface.center - surface_0.center),
            ),
            mode_parameters_on_surface_0=mode_parameters_right_after_surface_0,
        )
        local_mode_parameters_right_after_surface_2 = arm.propagate_local_mode_parameters()
        mode_parameters_after_surface_2 = local_mode_parameters_right_after_surface_2.to_mode_parameters(
            location_of_local_mode_parameter=arm.central_line.parameterization(t=arm.central_line.length),
            k_vector=arm.central_line.k_vector,  # ARBITRARY - ASSUMES CENTREAL LINE IS PERPENDICULAR TO SURFACE_2 - SHOULD BE CHANGED TO
            # THE NEXT CENTRAL LINE, AFTER REFRACTION
        )
        second_angle_of_incidence = calculate_incidence_angle(
            surface=second_surface,
            mode_parameters=mode_parameters_after_surface_2,
        )
        diff = first_angle_of_incidence - second_angle_of_incidence
        return diff

    R_1 = optimize.brentq(f=f_for_root, a=h, b=1000 * surface_0.radius)  # surface_0.radius

    second_surface = match_surface_to_radius(R_1)

    return second_surface


def find_required_value_for_desired_change(
    # This is the best function in the world <3
    cavity_generator: Callable,  # Takes a float as input and returns a cavity
    desired_parameter: Callable,  # Takes a cavity as input and returns a float
    # (NA of some arm, length of some arm, radius of curvature, etc.)
    desired_value: float,  # Desired value to end up with for the parameter
    solver: Callable = optimize.fsolve,
    print_progress=False,
    **kwargs,
) -> Cavity:
    def f_root(input_parameter_value: Union[float, np.ndarray]):
        if isinstance(input_parameter_value, np.ndarray):
            input_parameter_value = input_parameter_value[0]
        perturbed_cavity = cavity_generator(input_parameter_value)
        output_parameter_value = desired_parameter(perturbed_cavity)
        diff = output_parameter_value - desired_value
        if print_progress:
            print(
                f"input_parameter_value: {input_parameter_value:.10e}, output_parameter_value: {output_parameter_value:.3e}, diff: {diff:.3e}"
            )
        return diff

    perturbation_value = solver(f_root, **kwargs)
    cavity = cavity_generator(perturbation_value[0])
    return cavity


def find_required_perturbation_for_desired_change(
    cavity: Cavity,
    parameter_index_to_change: Tuple[int, int],
    desired_parameter: Callable,
    desired_value: float,
    **kwargs,
) -> Cavity:
    def cavity_generator(perturbation_value: float):
        return perturb_cavity(cavity, parameter_index_to_change, perturbation_value)

    return find_required_value_for_desired_change(cavity_generator, desired_parameter, desired_value, **kwargs)


def mirror_lens_mirror_cavity_generator(
    NA_left: float = 0.1,
    waist_to_lens: float = 5e-3,
    h: float = 3.875e-3,
    R_left: float = 5e-3,
    R_right: float = 5e-3,
    T_c: float = 5e-3,
    T_edge: float = 1e-3,
    right_arm_length: float = 0.3,
    lens_fixed_properties="fused_silica",
    mirrors_fixed_properties="fused_silica",
    symmetric_left_arm: bool = True,
    waist_to_left_mirror: float = 5e-3,
    lambda_0_laser=1064e-9,
    set_h_instead_of_w: bool = True,
    right_mirror_on_waist: bool = False,
    auto_set_right_arm_length: bool = True,
    set_R_right_to_equalize_angles: bool = True,
    set_R_right_to_R_left: bool = False,
    set_R_right_to_collimate: bool = False,
    set_R_left_to_collimate: bool = True,
    **kwargs,
):

    # This function receives many parameters that can define a cavity of mirror-lens-mirror and creates a Cavity object
    # out of them.
    assert not (
        set_R_left_to_collimate and set_R_right_to_collimate
    ), "Too many solutions: can't set automatically both R_left to collimate and R_right to collimate"
    assert (
        not (set_R_right_to_collimate + set_R_right_to_equalize_angles + set_R_right_to_R_left) > 1
    ), "Too many constraints on R_right"

    if set_R_left_to_collimate or set_R_right_to_collimate:
        if set_R_left_to_collimate:

            def cavity_generator(R_left_):
                cavity = mirror_lens_mirror_cavity_generator(
                    NA_left=NA_left,
                    waist_to_lens=waist_to_lens,
                    h=h,
                    R_left=R_left_,
                    R_right=R_right,
                    T_c=T_c,
                    T_edge=T_edge,
                    right_arm_length=right_arm_length,
                    lens_fixed_properties=lens_fixed_properties,
                    mirrors_fixed_properties=mirrors_fixed_properties,
                    symmetric_left_arm=symmetric_left_arm,
                    waist_to_left_mirror=waist_to_left_mirror,
                    lambda_0_laser=lambda_0_laser,
                    set_h_instead_of_w=set_h_instead_of_w,
                    auto_set_right_arm_length=auto_set_right_arm_length,
                    set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
                    set_R_right_to_R_left=set_R_right_to_R_left,
                    right_mirror_on_waist=right_mirror_on_waist,
                    set_R_right_to_collimate=False,
                    set_R_left_to_collimate=False,
                    **kwargs,
                )
                return cavity

            x0 = R_left
        else:

            def cavity_generator(R_right_):
                cavity = mirror_lens_mirror_cavity_generator(
                    NA_left=NA_left,
                    waist_to_lens=waist_to_lens,
                    h=h,
                    R_left=R_left,
                    R_right=R_right_,
                    T_c=T_c,
                    T_edge=T_edge,
                    right_arm_length=right_arm_length,
                    lens_fixed_properties=lens_fixed_properties,
                    mirrors_fixed_properties=mirrors_fixed_properties,
                    symmetric_left_arm=symmetric_left_arm,
                    waist_to_left_mirror=waist_to_left_mirror,
                    lambda_0_laser=lambda_0_laser,
                    set_h_instead_of_w=set_h_instead_of_w,
                    auto_set_right_arm_length=auto_set_right_arm_length,
                    set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
                    set_R_right_to_R_left=set_R_right_to_R_left,
                    right_mirror_on_waist=right_mirror_on_waist,
                    set_R_right_to_collimate=False,
                    set_R_left_to_collimate=False,
                    **kwargs,
                )
                return cavity

            x0 = R_right

        cavity = find_required_value_for_desired_change(
            cavity_generator=cavity_generator,
            # Takes a float as input and returns a cavity
            desired_parameter=lambda cavity: 1 / cavity.arms[2].mode_parameters_on_surfaces[0].z_minus_z_0[0],
            desired_value=-2 / right_arm_length,
            x0=x0,
        )
        return cavity

    if set_R_right_to_R_left:
        R_right = R_left

    mirrors_material_properties = convert_material_to_mirror_or_lens(
        PHYSICAL_SIZES_DICT[f"thermal_properties_{mirrors_fixed_properties}"], "mirror"
    )
    lens_material_properties = convert_material_to_mirror_or_lens(
        PHYSICAL_SIZES_DICT[f"thermal_properties_{lens_fixed_properties}"], "lens"
    )

    # Generate left arm's mirror:
    w_0_left = lambda_0_laser / (np.pi * NA_left)
    left_waist_x = 0
    if symmetric_left_arm:
        x_left = left_waist_x - waist_to_lens
    else:
        x_left = waist_to_left_mirror
    mode_left_center = np.array([left_waist_x, 0, 0])

    mode_left_k_vector = np.array([1, 0, 0])
    mode_left = ModeParameters(
        center=np.stack([mode_left_center, mode_left_center], axis=0),
        k_vector=mode_left_k_vector,
        w_0=np.array([w_0_left, w_0_left]),
        principle_axes=np.array([[0, 0, 1], [0, 1, 0]]),
        lambda_0_laser=lambda_0_laser,
    )
    mirror_left = match_a_mirror_to_mode(mode_left, x_left - mode_left.center[0, 0], mirrors_material_properties)
    # Generate lens:
    # if lens_material_properties_override:
    (
        n,
        alpha_lens,
        beta_lens,
        kappa_lens,
        dn_dT_lens,
        nu_lens,
        alpha_absorption_lens,
        intensity_reflectivity,
        intensity_transmittance,
        temperature,
    ) = lens_material_properties.to_array
    x_2_left = left_waist_x + waist_to_lens
    surface_left = CurvedRefractiveSurface(
        center=np.array([x_2_left, 0, 0]),
        radius=R_left,
        outwards_normal=np.array([-1, 0, 0]),
        n_1=1,
        n_2=n,
        curvature_sign=-1,
        name="lens_left",
        thermal_properties=lens_material_properties,
    )
    if set_R_right_to_equalize_angles:
        surface_right = find_equal_angles_surface(
            mode_before_lens=mode_left,
            surface_0=surface_left,
            T_edge=T_edge,
            h=h,
            lambda_0_laser=lambda_0_laser,
        )
        T_c = np.linalg.norm(surface_right.center - surface_left.center)
    else:
        if set_h_instead_of_w:
            # In case the user wants to set the height and the edge thickness of the lens instead of the thickness, then
            # the thickness is calculated using tha radii and the edge thickness.
            assert (
                R_left
            ), f"transverse radius of lens ({h:.2e}), can not be bigger than left radius of curvature ({R_left:.2e})"
            assert (
                R_right
            ), f"transverse radius of lens ({h:.2e}), can not be bigger than right radius of curvature ({R_right:.2e})"
            dT_c_left = R_left * (1 - np.sqrt(1 - h**2 / R_left**2))
            dT_c_right = R_right * (1 - np.sqrt(1 - h**2 / R_right**2))
            T_c = T_edge + dT_c_left + dT_c_right

        x_2_right = x_2_left + T_c

        surface_right = CurvedRefractiveSurface(
            center=np.array([x_2_right, 0, 0]),
            radius=R_right,
            outwards_normal=np.array([1, 0, 0]),
            n_1=n,
            n_2=1,
            curvature_sign=1,
            name="lens_right",
            thermal_properties=lens_material_properties,
        )

    mode_parameters_just_before_surface_left = mode_left.local_mode_parameters(
        np.linalg.norm(surface_left.center - mode_left.center[0])
    )

    mode_parameters_right_after_surface_left = propagate_local_mode_parameter_through_ABCD(
        mode_parameters_just_before_surface_left,
        surface_left.ABCD_matrix(cos_theta_incoming=1),
        n_1=1,
        n_2=n,
    )

    arm = Arm(
        surface_0=surface_left,
        surface_1=surface_right,
        central_line=Ray(
            origin=surface_left.center,
            k_vector=normalize_vector(surface_right.center - surface_left.center),
            length=np.linalg.norm(surface_right.center - surface_left.center),
        ),
        mode_parameters_on_surface_0=mode_parameters_right_after_surface_left,
    )
    mode_parameters_right_after_surface_right = arm.propagate_local_mode_parameters()

    mode_right = mode_parameters_right_after_surface_right.to_mode_parameters(
        location_of_local_mode_parameter=surface_right.center,
        k_vector=np.array([1, 0, 0]),
    )

    z_minus_z_0_right_surface = mode_parameters_right_after_surface_right.z_minus_z_0[0]

    if z_minus_z_0_right_surface > 0:
        z_minus_z_0_right_mirror = (
            z_minus_z_0_right_surface + right_arm_length + 1 / z_minus_z_0_right_surface
        )  # This 1 / z_minus_z_0_right_surface is here to make the mirror further as the NA grows larger.
    else:
        if auto_set_right_arm_length:
            z_minus_z_0_right_mirror = -z_minus_z_0_right_surface
        elif right_mirror_on_waist:
            z_minus_z_0_right_mirror = 0
        else:
            z_minus_z_0_right_mirror = z_minus_z_0_right_surface + right_arm_length
    mirror_right = match_a_mirror_to_mode(mode_right, z_minus_z_0_right_mirror, mirrors_material_properties)
    mirror_left_params = mirror_left.to_params
    mirror_right_params = mirror_right.to_params

    params_lens = surface_right.to_params
    params_lens.x = (surface_left.center[0] + surface_right.center[0]) / 2
    params_lens.r_1 = surface_left.radius
    params_lens.r_2 = surface_right.radius
    params_lens.T_c = T_c
    params_lens.n_inside_or_after = n
    params_lens.n_outside_or_before = 1
    params_lens.surface_type = SurfacesTypes.thick_lens

    params = [mirror_left_params, params_lens, mirror_right_params]

    cavity = Cavity.from_params(
        params,
        lambda_0_laser=lambda_0_laser,
        standing_wave=True,
        p_is_trivial=True,
        t_is_trivial=True,
        set_mode_parameters=True,
        names=["Left mirror", "lens_left", "lens_right", "Right mirror"],
        initial_mode_parameters=mode_left,
        **kwargs,
    )

    return cavity


def plot_mirror_lens_mirror_cavity_analysis(
    cavity: Cavity,
    auto_set_x: bool = True,
    x_span: float = 4e-1,
    auto_set_y: bool = True,
    y_span: float = 8e-3,
    camera_center: Union[int, float] = 1,
    add_unheated_cavity: bool = False,
    minimal_h_divided_by_spot_size: float = 2.5,
    T_edge=1e-3,
    CA: float = 5e-3,
    h: float = 3.875e-3,
    set_h_instead_of_w: bool = True,
):
    # Assumes: surfaces[0] is the left mirror, surfaces[1] is the lens_left side, surfaces[2] is the lens_right side,
    # surfaces[3] is the right mirror.
    R_left = cavity.surfaces[1].radius
    R_right = cavity.surfaces[2].radius
    T_c = np.linalg.norm(cavity.surfaces[2].center - cavity.surfaces[1].center)
    angle_right = cavity.arms[2].calculate_incidence_angle(surface_index=0)
    angle_left = cavity.arms[0].calculate_incidence_angle(surface_index=1)
    spot_size_lens_right = cavity.arms[1].mode_parameters_on_surfaces[1].spot_size[0]
    CA_divided_by_2spot_size = CA / (2 * spot_size_lens_right)
    short_arm_NA = cavity.arms[0].mode_parameters.NA[0]
    long_arm_NA = cavity.arms[2].mode_parameters.NA[0]
    short_arm_length = np.linalg.norm(cavity.surfaces[1].center - cavity.surfaces[0].center)
    long_arm_length = np.linalg.norm(cavity.surfaces[3].center - cavity.surfaces[2].center)
    waist_to_lens_short_arm = cavity.surfaces[1].center[0] - cavity.mode_parameters[0].center[0, 0]
    waist_to_lens_long_arm = cavity.mode_parameters[2].center[0, 0] - cavity.surfaces[2].center[0]
    spot_size_left_mirror = cavity.arms[0].mode_parameters_on_surfaces[0].spot_size[0]
    spot_size_right_mirror = cavity.arms[2].mode_parameters_on_surfaces[1].spot_size[0]
    R_left_mirror = cavity.surfaces[0].radius
    x_left_mirror = cavity.surfaces[0].center[0]
    x_lens_right = cavity.surfaces[2].center[0]

    if add_unheated_cavity:
        fig, ax = plt.subplots(2, 1, figsize=(16, 12))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        ax = [ax]
    cavity.plot(axis_span=x_span, camera_center=camera_center, ax=ax[0])

    minimal_width_lens = find_minimal_width_for_spot_size_and_radius(
        radius=R_left,
        spot_size_radius=spot_size_lens_right,
        T_edge=T_edge,
        h_divided_by_spot_size=minimal_h_divided_by_spot_size,
    )
    geometric_feasibility = True

    if set_h_instead_of_w:
        if CA_divided_by_2spot_size < 2.5:
            geometric_feasibility = False
        lens_specs_string = (
            f"R_left = {R_left:.3e},  R_right = {R_right:.3e},  D = {2 * h:.3e},  T_edge = {T_edge:.2e},  T_c = {T_c:.3e},  CA={CA:.2e}\n"
            f"spot_size (2w) = {2 * spot_size_lens_right:.3e},   CA / 2w_spot_size = {CA_divided_by_2spot_size:.3e}, lens is wide enough = {geometric_feasibility},   {angle_left=:.2f},   {angle_right=:.2f}"
        )
    else:
        if T_c < minimal_width_lens:
            geometric_feasibility = False
        minimal_CA_lens = minimal_h_divided_by_spot_size * spot_size_lens_right
        lens_specs_string = f"R_lens = {R_left:.3e},  T_c = {T_c:.3e},  minimal_w_lens = {minimal_width_lens:.2e},  minimal_CA_lens={minimal_CA_lens:.3e},  lens is thick enough = {geometric_feasibility}"

    ax[0].set_title(
        f"Short Arm: NA = {short_arm_NA:.3e},  length = {short_arm_length:.3e} [m], waist to lens = {waist_to_lens_short_arm:.3e}\n"  #
        f" Long Arm NA = {long_arm_NA:.3e},  length = {long_arm_length:.3e} [m], waist to lens = {waist_to_lens_long_arm:.3e}\n"  #
        f"{lens_specs_string}, \n"
        f"R_left_mirror = {R_left_mirror:.3e}, spot diameters left mirror = {2 * spot_size_left_mirror:.2e}, R_right = {cavity.surfaces[3].radius:.3e}, spot diameters right mirror = {2 * spot_size_right_mirror:.2e}"
    )

    if auto_set_x:
        # cavity_length = cavity.surfaces[3].center[0] - cavity.surfaces[0].center[0]
        # ax[0].set_xlim(cavity.surfaces[0].center[0] - 0.01 * cavity_length, cavity.surfaces[3].center[0] + 0.01 * cavity_length)
        ax[0].set_xlim(x_left_mirror - 0.01, x_lens_right + 0.35)
    if auto_set_y:
        y_lim = maximal_lens_height(R_left, T_c) * 1.1
    else:
        y_lim = y_span
    ax[0].set_ylim(-y_lim, y_lim)
    if add_unheated_cavity:
        unheated_cavity = cavity.thermal_transformation()
        unheated_cavity.plot(axis_span=x_span, camera_center=camera_center, ax=ax[1])
        ax[1].set_xlim(ax[0].get_xlim())
        ax[1].set_ylim(ax[0].get_ylim())
        ax[1].set_title(
            f"unheated_cavity, short arm NA={short_arm_NA:.2e}, Left mirror 2*spot size = {2 * spot_size_left_mirror:.2e}"
        )
    plt.subplots_adjust(hspace=0.35)
    fig.tight_layout()

# def decompose_a_thick_lens_single_surfaces(cavity: Cavity) -> Cavity:
#     new_params = [s.to_params for ]