import copy

import numpy as np
from typing import List, Tuple, Optional, Union, Callable, Any
import pickle as pkl
import warnings
# import dataclass:
from dataclasses import dataclass

# %% Constants
CENTRAL_LINE_TOLERANCE = 1
STRETCH_FACTOR = 1  # 0.001
C_LIGHT_SPEED = 299792458  # [m/s]
ROOM_TEMPERATURE = 293  # [K]
LAMBDA_0_LASER = 1064e-9
LAMBDA_1_LASER = 532e-9
EPSILON_0_PERMITTIVITY = 8.854187e-12  # [F/m]
MU_0_PERMEABILITY = 1.256637e-6

# Every optical element has a np.ndarray representation, and those two dictionaries defines the order and meaning of
# the array columns.
PRETTY_INDICES_NAMES = {'x': 'x [m]',
                        'y': 'y [m]',
                        'theta': 'Elevation angle [rads]',
                        'phi': 'Azimuthal angle [rads]',
                        'r_1': 'Radius of curvature 1 [m]',
                        'r_2': 'Radius of curvature 2 [m]',
                        'n_outside_or_before': 'Index of refraction (before the surface)',
                        'T_c': 'Center thickness [m]',
                        'n_inside_or_after': 'Index of refraction (after the surface)',
                        'z': 'z [m]',
                        'curvature_sign': 'Curvature sign',
                        'material_refractive_index': 'Material refractive index',
                        'alpha_expansion': 'Thermal expansion coefficient [1/K]',
                        'beta_surface_absorption': 'Surface power absorption coefficient',
                        'kappa_conductivity': 'Thermal conductivity coefficient [W/(m*K)]',
                        'dn_dT': 'dn_dT [1/K]',
                        'nu_poisson_ratio': 'Poisson ratio',
                        'alpha_volume_absorption': 'Volume power absorption coefficient [1/m]',
                        'intensity_reflectivity': 'Intensity reflectivity',
                        'intensity_transmittance': 'Transmissivity',
                        'temperature': 'Temperature increase [K]',
                        'surface_type': 'Surface type'}
INDICES_DICT = {name: i for i, name in enumerate(PRETTY_INDICES_NAMES.keys())}

INDICES_DICT_INVERSE = {v: k for k, v in INDICES_DICT.items()}
# set numpy to raise an error on warnings:
SURFACE_TYPES_DICT = {'curved_mirror': 0, 'thick_lens': 1, 'curved_refractive_surface': 2, 'ideal_lens': 3,
                      'flat_mirror': 4, 'ideal_thick_lens': 5}
SURFACE_TYPES_DICT_INVERSE = {v: k for k, v in SURFACE_TYPES_DICT.items()}

@dataclass
class ParamsNames:
    x: str = 'x'
    y: str = 'y'
    theta: str = 'theta'
    phi: str = 'phi'
    r_1: str = 'r_1'
    r_2: str = 'r_2'
    n_outside_or_before: str = 'n_outside_or_before'
    T_c: str = 'T_c'
    n_inside_or_after: str = 'n_inside_or_after'
    z: str = 'z'
    curvature_sign: str = 'curvature_sign'
    material_refractive_index: str = 'material_refractive_index'
    alpha_expansion: str = 'alpha_expansion'
    beta_surface_absorption: str = 'beta_surface_absorption'
    kappa_conductivity: str = 'kappa_conductivity'
    dn_dT: str = 'dn_dT'
    nu_poisson_ratio: str = 'nu_poisson_ratio'
    alpha_volume_absorption: str = 'alpha_volume_absorption'
    intensity_reflectivity: str = 'intensity_reflectivity'
    intensity_transmittance: str = 'intensity_transmittance'
    temperature: str = 'temperature'
    surface_type: str = 'surface_type'


@dataclass
class SurfacesTypes:
    thick_lens = 'thick_lens'
    curved_mirror = 'curved_mirror'
    curved_refractive_surface = 'curved_refractive_surface'
    ideal_lens = 'ideal_lens'
    flat_mirror = 'flat_mirror'
    ideal_thick_lens = 'ideal_thick_lens'
    flat_surface = 'flat_surface'  # Not an optical element, just a helper for the central line.

    @staticmethod
    def from_integer_representation(integer_representation: int) -> str:
        return SurfacesTypes.__dict__[SURFACE_TYPES_DICT_INVERSE[integer_representation]]

    @staticmethod
    def has_refractive_index(surface_type: str) -> bool:
        return surface_type in [SurfacesTypes.curved_refractive_surface,
                                SurfacesTypes.thick_lens,
                                SurfacesTypes.ideal_thick_lens]

@dataclass
class CurvatureSigns:
    convex = -1
    concave = 1

    @staticmethod
    def from_integer_representation(integer_representation: int) -> int:
        if integer_representation == 1:
            return CurvatureSigns.convex
        elif integer_representation == -1:
            return CurvatureSigns.concave
        else:
            raise ValueError("Curvature sign must be either 1 or -1")

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
    "material_properties_sapphire": MaterialProperties(
        refractive_index=1.76,
        alpha_expansion=5.5e-6,  # https://www.shinkosha.com/english/techinfo/feature/thermal-properties-of-sapphire/#:~:text=Sapphire%20has%20a%20large%20linear,very%20resistant%20to%20thermal%20shock., https://www.roditi.com/SingleCrystal/Sapphire/Properties.html
        beta_surface_absorption=1e-6,  # DUMMY
        kappa_conductivity=46.06,  # https://www.google.com/search?q=sapphire+thermal+conductivity&rlz=1C1GCEB_enIL1023IL1023&oq=sapphire+thermal+c&aqs=chrome.0.35i39i650j69i57j0i20i263i512j0i22i30l3j0i10i15i22i30j0i22i30l3.3822j0j1&sourceid=chrome&ie=UTF-8, https://www.shinkosha.com/english/techinfo/feature/thermal-properties-of-sapphire/, https://www.shinkosha.com/english/techinfo/feature/thermal-properties-of-sapphire/
        dn_dT=11.7e-6,  # https://secwww.jhuapl.edu/techdigest/Content/techdigest/pdf/V14-N01/14-01-Lange.pdf
        nu_poisson_ratio=0.3,  # https://www.google.com/search?q=sapphire+poisson+ratio&rlz=1C1GCEB_enIL1023IL1023&sxsrf=AB5stBgEUZwh7l9RzN9GwxjMPCw_DcShAw%3A1688647440018&ei=ELemZI1h0-2SBaukk-AH&ved=0ahUKEwiNqcD2jfr_AhXTtqQKHSvSBHwQ4dUDCA8&uact=5&oq=sapphire+poisson+ratio&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIECAAQHjIICAAQigUQhgMyCAgAEIoFEIYDMggIABCKBRCGAzIICAAQigUQhgMyCAgAEIoFEIYDOgoIABBHENYEELADSgQIQRgAUJsFWJsFYNQJaAFwAXgAgAF5iAF5kgEDMC4xmAEAoAEBwAEByAEI&sclient=gws-wiz-serp
        alpha_volume_absorption=100e-6 * 100,  # The data is in ppm/cm and I convert it to ppm/m, hence the "*100".  # https://labcit.ligo.caltech.edu/~ligo2/pdf/Gustafson2c.pdf  # https://www.nature.com/articles/s41598-020-80313-1  # https://www.crystran.co.uk/optical-materials/sapphire-al2o3,
        intensity_reflectivity=100e-6,  # DUMMY - for lenses
        intensity_transmittance=1 - 100e-6 - 1e-6,
    ),  # DUMMY - for lenses
    "material_properties_ULE": MaterialProperties(
        alpha_expansion=7.5e-8,  # https://en.wikipedia.org/wiki/Ultra_low_expansion_glass#:~:text=It%20has%20a%20thermal%20conductivity,C%20%5B1832%20%C2%B0F%5D, https://www.corning.com/media/worldwide/csm/documents/7972%20ULE%20Product%20Information%20Jan%202016.pdf
        kappa_conductivity=1.31,
        nu_poisson_ratio=0.17,
        beta_surface_absorption=1e-6,  # DUMMY
        intensity_reflectivity=1 - 100e-6 - 1e-6 - 10e-6,  # All - transmittance - absorption - scattering
        intensity_transmittance=100e-6,  # DUMMY - for mirrors
    ),
    "material_properties_fused_silica": MaterialProperties(  # https://www.corning.com/media/worldwide/csm/documents/HPFS_Product_Brochure_All_Grades_2015_07_21.pdf
        refractive_index=1.45,  # The next link is ignored due to the link above: https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson,
        alpha_expansion=0.52e-6,  # The next link is ignored due to the link above: https://www.rp-photonics.com/fused_silica.html#:~:text=However%2C%20fused%20silica%20may%20exhibit,10%E2%88%926%20K%E2%88%921., https://www.swiftglass.com/blog/material-month-fused-silica/
        beta_surface_absorption=1e-6,  # DUMMY
        kappa_conductivity=1.38,  # This link and the link above agree: https://www.swiftglass.com/blog/material-month-fused-silica/, https://www.heraeus-conamic.com/knowlegde-base/properties
        dn_dT=12e-6,  # https://iopscience.iop.org/article/10.1088/0022-3727/16/5/002/pdf
        nu_poisson_ratio=0.16,
        alpha_volume_absorption=1e-3,  # https://www.crystran.co.uk/optical-materials/silica-glass-sio2
        intensity_reflectivity=100e-6,  # DUMMY - for lenses
        intensity_transmittance=1 - 100e-6 - 1e-6,  # DUMMY - for lenses
    ),  # https://www.azom.com/properties.aspx?ArticleID=1387),
    "material_properties_yages_fused_silica": MaterialProperties(
        refractive_index=1.81,
        alpha_expansion=8e-6,  # https://www.crystran.co.uk/optical-materials/yttrium-aluminium-garnet-yag
        beta_surface_absorption=1e-6,  # DUMMY
        kappa_conductivity=11.2,  # https://www.scientificmaterials.com/downloads/Nd_YAG.pdf, This does not agree: https://pubs.aip.org/aip/jap/article/131/2/020902/2836262/Thermal-conductivity-and-management-in-laser-gain
        dn_dT=9e-6,  # https://pubmed.ncbi.nlm.nih.gov/18319922/
        nu_poisson_ratio=0.25,  #  https://www.crystran.co.uk/userfiles/files/yttrium-aluminium-garnet-yag-data-sheet.pdf, https://www.korth.de/en/materials/detail/YAG
    ),
    "material_properties_bk7": MaterialProperties(
        alpha_expansion=7.1e-6,  # https://www.pgo-online.com/intl/BK7.html
        alpha_volume_absorption = 0.001285 * 100,  # The data is in 1/cm and I convert it to 1/m https://refractiveindex.info/?shelf=3d&book=glass&page=BK7
        kappa_conductivity=1.114,  # https://www.pgo-online.com/intl/BK7.html
        refractive_index=1.507,  # https://www.pgo-online.com/intl/BK7.html
        nu_poisson_ratio=0.206,  # https://www.matweb.com/search/DataSheet.aspx?MatGUID=a8c2d05c7a6244399bbc2e15c0438cb9, https://www.pgo-online.com/intl/BK7.html
        intensity_reflectivity=1 - 100e-6 - 1e-6 - 10e-6,  # All - transmittance - absorption - scattering
        intensity_transmittance=100e-6,  # DUMMY - for mirrors
        beta_surface_absorption=1e-6,  # DUMMY
    ),
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
    name: Optional[str]
    surface_type: str
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
    diameter: float = np.nan  # diameter of the optical element, None if not specified.

    # def __post_init__(self):
    #     assert self.material_properties.refractive_index == self.n_inside_or_after or self.material_properties.refractive_index == self.n_outside_or_before or np.isnan(self.material_properties.refractive_index), "The refractive index of the material properties is neither of the refractive indices of the optical element!"

    def __repr__(self):
        surface_type_string = f"'{self.surface_type}'"
        surface_type_string = surface_type_string.ljust(33)
        name_string = f"'{self.name}'"
        name_string = name_string.ljust(25)
        curvature_sign_string = "CurvatureSigns.concave" if self.curvature_sign == 1 else "CurvatureSigns.convex"
        return (
            f"OpticalElementParams("
            f"name={name_string},"
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
            f"diameter={pretty_print_number(self.diameter)}, "
            f"material_properties={self.material_properties})"
        )

    @property
    def to_array(self) -> np.ndarray:
        # loses the name information, as it is string.
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
                INDICES_DICT["diameter"],
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
            self.diameter,
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
            name=None,  # This is not saved in the array
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
            diameter=params[INDICES_DICT["diameter"]],
            material_properties=material_properties,
        )


@dataclass
class PerturbationPointer:
    element_index: int
    parameter_name: str
    perturbation_value: Optional[Union[float, int, np.ndarray]] = None

    def __getitem__(self, index):
        if self.perturbation_value is None or isinstance(self.perturbation_value, (float, int)):
            return self
        else:
            return PerturbationPointer(
                element_index=self.element_index,
                parameter_name=self.parameter_name,
                perturbation_value=self.perturbation_value[index]
            )

    def __iter__(self):
        if self.perturbation_value is None or np.isscalar(self.perturbation_value):
            yield self
        else:
            for value in self.perturbation_value:
                yield PerturbationPointer(
                    element_index=self.element_index,
                    parameter_name=self.parameter_name,
                    perturbation_value=value
                )

    def __call__(self, perturbation_value: float):
        return PerturbationPointer(
            element_index=self.element_index,
            parameter_name=self.parameter_name,
            perturbation_value=perturbation_value
        )

    def __len__(self):
        if self.perturbation_value is None or isinstance(self.perturbation_value, (float, int)):
            return 1
        else:
            return len(self.perturbation_value)

    def apply_to_params(self, params: List[OpticalElementParams]):
        # Changes the original params
        current_value = getattr(
            params[self.element_index], self.parameter_name
        )
        new_value = current_value + self.perturbation_value
        setattr(
            params[self.element_index], self.parameter_name, new_value
        )


def pretty_print_array(array: np.ndarray):
    # Prints an array in a way that can be copy-pasted into the code.
    print(f"np.{array=}".replace('array=', "").replace("nan", "np.nan").replace("\n", "").replace("],", "],\n").replace(",        ", ", ").replace(",      ", ", ").replace("       [", "          ["))


def pretty_print_number(number: Optional[float], represents_angle: bool = False):
    if number is None:
        pre_padded = 'None'
    elif np.isnan(number):
        pre_padded = 'np.nan'
    elif number == 0:
        pre_padded = '0'
    else:
        if represents_angle:
            number /= np.pi
        formatted = f"{number:.15e}"
        # Remove trailing zeros and the decimal point if not necessary
        parts = formatted.split('e')
        parts[0] = parts[0].rstrip('0').rstrip('.')
        pre_padded = 'e'.join(parts)
        if represents_angle:
            pre_padded += ' * np.pi'
    final_string = pre_padded.ljust(24, ' ')
    return final_string


def nvl(var, val: Any = np.nan):
    if var is None:
        return val
    return var


def plane_name_to_xy_indices(plane: str) -> Tuple[int, int]:
    if plane in ['xy', 'yx']:
        x_index = 0
        y_index = 1
    elif plane in ['xz', 'zx']:
        x_index = 0
        y_index = 2
    elif plane in ['yz', 'zy']:
        x_index = 1
        y_index = 2
    else:
        raise ValueError("plane must be one of 'xy', 'xz', 'yz'")
    return x_index, y_index


def maximal_lens_height(R: float, w: float) -> float:
    return R * np.sqrt(1 - ((R - w/2) / R) ** 2)


def safe_exponent(a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    a_safe = np.clip(a=np.real(a), a_min=-200, a_max=None) + 1j * np.imag(a)  # ARBITRARY
    return np.exp(a_safe)


def ABCD_free_space(length: float) -> np.ndarray:
    return np.array([[1, length, 0, 0],
                     [0, 1,      0, 0],
                     [0, 0,      1, length],  # / cos_theta_between_planes
                     [0, 0,      0, 1]])


def normalize_vector(vector: Union[np.ndarray, list], ignore_null_vectors: bool = False) -> np.ndarray:
    if isinstance(vector, list):
        vector = np.array(vector)
    norm = np.linalg.norm(vector, axis=-1)[..., np.newaxis]
    if not ignore_null_vectors:
        return vector / norm
    else:
        norm = np.where(norm == 0, 1, norm)
        return vector / norm


def rotation_matrix_around_n(n, theta):
    # Rotates a vector around the axis n by theta radians.
    # This funny stacked syntax is to allow theta to be of any dimension
    A = np.stack([np.stack([np.cos(theta) + n[0] ** 2 * (1 - np.cos(theta)),
                            n[0] * n[1] * (1 - np.cos(theta)) - n[2] * np.sin(theta),
                            n[0] * n[2] * (1 - np.cos(theta)) + n[1] * np.sin(theta)], axis=-1),
                  np.stack([n[1] * n[0] * (1 - np.cos(theta)) + n[2] * np.sin(theta),
                            np.cos(theta) + n[1] ** 2 * (1 - np.cos(theta)),
                            n[1] * n[2] * (1 - np.cos(theta)) - n[0] * np.sin(theta)], axis=-1),
                  np.stack([n[2] * n[0] * (1 - np.cos(theta)) - n[1] * np.sin(theta),
                            n[2] * n[1] * (1 - np.cos(theta)) + n[0] * np.sin(theta),
                            np.cos(theta) + n[2] ** 2 * (1 - np.cos(theta))], axis=-1)], axis=-1)
    return A


def focal_length_of_lens(R_1, R_2, n, width):
    # for R_1 = R_2 = 0.05, d=0.01, n=1.5, this function returns 1.5 [m]
    one_over_f = (n - 1) * ((1 / R_1) + (1 / R_2) - ((n - 1) * width) / (n * R_1 * R_2))
    return 1 / one_over_f


def sin_without_trailing_epsilon(phi: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    pi_multiplications = np.mod(phi, np.pi)  # This is to avoid numerical trailing epsilon.
    sin_phi = np.sin(phi)
    if isinstance(sin_phi, (float, int)):
        if pi_multiplications == 0:
            sin_phi = 0
    else:
        sin_phi[pi_multiplications == 0] = 0
    return sin_phi


def cos_without_trailing_epsilon(phi: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    pi_half_multiplications = np.mod(phi - np.pi / 2, np.pi)  # This is to avoid numerical trailing epsilon.
    cos_phi = np.cos(phi)
    if isinstance(cos_phi, (float, int)):
        if pi_half_multiplications == 0:
            cos_phi = 0
    else:
        cos_phi[pi_half_multiplications == 0] = 0
    return cos_phi


def unit_vector_of_angles(theta: Union[np.ndarray, float], phi: Union[np.ndarray, float]) -> np.ndarray:
    # Those are the angles of the unit vector in spherical coordinates, with respect to the global system of coordinates
    # theta and phi are assumed to be in radians.
    # Since I work mostly in the x-y plane, theta is measured from the x-y plane, so that it is usually close to 0.
    sin_phi = sin_without_trailing_epsilon(phi)
    cos_phi = cos_without_trailing_epsilon(phi)
    sin_theta = sin_without_trailing_epsilon(theta)
    cos_theta = cos_without_trailing_epsilon(theta)

    return np.stack([cos_theta * cos_phi, cos_theta * sin_phi, sin_theta], axis=-1)


def angles_of_unit_vector(unit_vector: Union[np.ndarray, float]) -> Union[Tuple[np.ndarray, np.ndarray],
                                                                          Tuple[float, float]]:
    # theta and phi are returned in radians
    theta = np.arcsin(unit_vector[..., 2])
    phi = np.arctan2(unit_vector[..., 1], unit_vector[..., 0])
    return theta, phi


def angles_distance(direction_vector_1: np.ndarray, direction_vector_2: np.ndarray):
    inner_product = np.sum(direction_vector_1 * direction_vector_2, axis=-1)
    inner_product = np.clip(inner_product, -1, 1)
    return np.arccos(inner_product)


def angles_difference(angle_1: Union[np.ndarray, float], angle_2: Union[np.ndarray, float]) -> np.ndarray:
    diff = angle_2 - angle_1
    result = np.mod(diff + np.pi, 2 * np.pi) - np.pi
    return result


def radius_of_f_and_n(f: float, n: float) -> float:
    return 2 * f * (n - 1)


def w_0_of_z_R(z_R: np.ndarray, lambda_0_laser: float, n: float) -> np.ndarray:
    # z_R_reduced is an array because of two transverse dimensions
    return np.sqrt(z_R * lambda_0_laser / (np.pi * n ** 2))


def z_R_of_w_0(w_0: np.ndarray, lambda_laser: float) -> np.ndarray:
    # lambda_laser is the wavelength of the laser in the medium = lambda_0 / n
    return np.pi * w_0 ** 2 / lambda_laser


def w_0_of_NA(NA: float, lambda_laser: float):
    return lambda_laser / (np.pi * NA)


def spot_size(z: np.ndarray, z_R: np.ndarray, lambda_0_laser: float, n: float) -> np.ndarray:  # AKA w(z)
    # lambda_laser is the wavelength of the laser in the medium = lambda_0 / n
    w_0 = w_0_of_z_R(z_R, lambda_0_laser, n)
    w_z = w_0 * np.sqrt(1 + (z / z_R) ** 2)
    return w_z


def stack_df_for_print(df):
    stacked_df = df.stack().reset_index()
    stacked_df.columns = ['Parameter', 'Element', 'Value']
    stacked_df = stacked_df[['Element', 'Parameter', 'Value']]
    return stacked_df


def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def gaussian_integral_2d_log(A: np.ndarray, b: np.ndarray, c):
    # The integral over exp( x.T A_2 x + b.T x + c):
    eigen_values = np.linalg.eigvals(A)
    A_inv = np.linalg.inv(A)
    dim = A.shape[0]
    try:
        log_integral = np.log(np.sqrt((2 * np.pi) ** dim / np.linalg.det(A))) + 0.5 * b.T @ A_inv @ b + c
    except FloatingPointError:
        log_integral = np.nan
    return log_integral


def gaussian_norm_log(A: np.ndarray, b: np.ndarray, c: float):
    return 1 / 2 * gaussian_integral_2d_log(A + np.conjugate(A), b + np.conjugate(b), c + np.conjugate(c))


def gaussians_overlap_integral(A_1: np.ndarray, A_2: np.ndarray,
                               # mu_1: np.ndarray, mu_2: np.ndarray, # Seems like I don't need the mus.
                               b_1: np.ndarray, b_2: np.ndarray,
                               c_1: float, c_2: float) -> float:
    A_1_conjugate = np.conjugate(A_1)
    b_1_conjugate = np.conjugate(b_1)
    c_1_conjugate = np.conjugate(c_1)

    # Plot a surface of the real part of the first gaussian, with x and y axes being the two dimensions, and z being the value of the gaussian at that point:
    # generate x and y as ranging from -5 sigmas to 5 sigmas of the first gaussian:

    A = A_1_conjugate + A_2
    b = b_1_conjugate + b_2
    c = c_1_conjugate + c_2
    # b = mu_1.T @ A_1_conjugate + mu_2.T @ A_2 + b_1_conjugate + b_2
    # c = (-1/2) * (mu_1.T @ A_1_conjugate @ mu_1 + mu_2.T @ A_2 @ mu_2) + c_1_conjugate + c_2
    normalization_factor_log = gaussian_norm_log(A_1, b_1, c_1) + gaussian_norm_log(A_2, b_2, c_2)
    integral_normalized_log = gaussian_integral_2d_log(A, b, c) - normalization_factor_log
    return safe_exponent(integral_normalized_log)


def interval_parameterization(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def functions_first_crossing(f: Callable, initial_step: float, crossing_value: float = 0.9,
                             accuracy: float = 0.001, max_f_eval: int = 200) -> float:
    # assumes f(0) == 1 and a decreasing function.
    stopping_flag = False
    increasing_ratio = 2
    n = 10
    last_n_evaluations = np.zeros(n)
    last_n_xs = np.zeros(n)
    borders_min = 0
    borders_max = np.nan
    borders_max_real = np.nan
    loop_counter = 0
    f_0 = f(0)
    if np.isnan(f_0):
        # warnings.warn('Function has no value at x_input=0, returning nan')
        f(0)
        warnings.warn('Function has no value at x_input=0, returning nan')
        return np.nan
    f_borders_min = f_0
    f_borders_max = np.nan
    last_n_evaluations[0] = f_0
    x = initial_step
    while not stopping_flag:
        loop_counter += 1
        f_x = f(x)
        last_n_xs[np.mod(loop_counter, n)] = x
        last_n_evaluations[np.mod(loop_counter, n)] = f_x
        if  loop_counter == max_f_eval and not np.isnan(borders_max_real):  # if it wasn't found but we know it's value
            # approximately, then interpolate it:
            x = (crossing_value - f_borders_min) / (f_borders_max - f_borders_min) * (
                    borders_max_real - borders_min) + borders_min
            # warnings.warn(
            #     f"Did not find crossing value, interpolated it between ({borders_min:.8e}, {f_borders_min:.5e}) and ({borders_max:.8e}, {f_borders_max:.5e}) to be ({x:.8e}, {crossing_value:.5e})")
            stopping_flag = True
        elif not np.any(np.invert(
                np.abs(1-last_n_evaluations) < 1e-18)) or loop_counter > max_f_eval:  # if the function is not
            # decreasing or we reached the max number of function evaluations:
            x = np.nan
            stopping_flag = True
        elif f_x > crossing_value + accuracy:  # If we are still above the crossing value (for x values too small):
            borders_min = x
            f_borders_min = f_x
            if np.isnan(borders_max):
                x *= increasing_ratio
            else:
                x = interval_parameterization(borders_min, borders_max, 0.5)
        elif f_x < crossing_value - accuracy:  # If we are below the crossing value (for x values too large, but still
            # not diverging f):
            increasing_ratio = 1.001  # Slow down steps
            borders_max = x
            borders_max_real = x
            f_borders_max = f_x
            x = interval_parameterization(borders_min, borders_max, 0.5)
        elif np.isnan(f_x):  # If x is too large and f diverges:
            increasing_ratio = 1.001
            borders_max = x
            # randomize new x with higher probability to be closer to borders_min (the pdf is phi(x)=2(1-x)), the cdf
            # is F(x)=2(x-x^2) and the inverse cdf is F^-1(y)=1-sqrt(1-y)
            y = np.random.uniform()
            x_normalized = 1 - np.sqrt(1 - y)
            x = interval_parameterization(borders_min, borders_max, x_normalized)
        elif not np.any(np.invert(np.isnan(last_n_evaluations))):  # If all the last n evaluations were nan:
            borders_max = np.min(last_n_xs)  # Set borders_max to be the first value from which
            # onwards it seems to always be nan.
            x = interval_parameterization(borders_min, borders_max, 0.5)
        elif crossing_value - accuracy < f_x < crossing_value + accuracy:
            stopping_flag = True
        else:
            raise ValueError('This should not happen')
    return x


def dT_c_of_a_lens(R, h):
    dT_c = R * (1- np.sqrt(1 - h ** 2 / R ** 2))
    return dT_c


def thick_lens_focal_length(T_c: float, R_1: float, R_2: float, n: float) -> float:
    f_inverse = (n - 1) * (1 / R_1 - 1 / R_2 + (n - 1) * T_c / (n * R_1 * R_2))
    f = 1 / f_inverse
    return f


def working_distance_of_a_lens(R_1, R_2, n, T_c):
    f = thick_lens_focal_length(T_c, R_1, R_2, n)
    h_2 = -f * (n-1) * T_c / (n * R_1)
    working_distance = f - h_2
    return working_distance


def stable_sqrt(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    # like square root, but returns nan if the parts of the input is negative, instead of throwing a FloatingPointError.
    if isinstance(x, np.ndarray):
        x[np.isnan(x)] = -1  # If it is already nan, then it will stay nan.
        s = np.sqrt(x + 0j)
        s[np.imag(s) != 0] = np.nan
        s = np.real(s)
        return s
    else:
        if np.isnan(x) or x < 0:
            return np.nan
        else:
            return np.sqrt(x)

def widget_convenient_exponent(x, base=10, scale=-10):
    y = base ** (x + scale) - base ** (-x + scale)
    return y

def widget_convenient_exponent_inverse(y, base=10.0, scale=-10.0):
    x = np.log((y * base ** - scale + np.sqrt(y**2 * base ** (-2*scale) + 4)) / 2) / np.log(base)
    return x


def generate_initial_parameters_grid(center: np.ndarray,
                                     range_limit: float,
                                     N_resolution: int,
                                     p_is_trivial: bool,
                                     t_is_trivial: bool):
    base_grid = np.linspace(-range_limit, range_limit, N_resolution)
    angle_factor = 100
    if p_is_trivial or t_is_trivial:
        if p_is_trivial:
            POS, ANGLE = np.meshgrid(base_grid, base_grid * angle_factor + center[1], indexing='ij')
            TRIVIAL_GRID = np.zeros_like(POS)
            initial_parameters = np.stack([POS + center[0], ANGLE + center[1], TRIVIAL_GRID + center[2], TRIVIAL_GRID + center[3]], axis=-1)
        else:  # (if theta is trivial)
            POS, ANGLE = np.meshgrid(base_grid, base_grid * angle_factor, indexing='ij')
            TRIVIAL_GRID = np.zeros_like(POS)
            initial_parameters = np.stack([TRIVIAL_GRID + center[0], TRIVIAL_GRID + center[1], POS + center[2], ANGLE + center[3]], axis=-1)
    else:
        X, T, Y, P = np.meshgrid(base_grid + center[0],
                                 base_grid * angle_factor + center[1],
                                 base_grid + center[2],
                                 base_grid * angle_factor + center[3], indexing='ij')
        initial_parameters = np.stack([X, T, Y, P], axis=-1)
    return initial_parameters

# %% Wave equations functions:
def green_function_free_space(r_source: np.ndarray, r_observer: np.ndarray, k: float) -> complex:
    r_vector = r_observer - r_source
    r = np.linalg.norm(r_vector, axis=-1)
    G = np.exp(1j * k * r) / (4 * np.pi * r)
    return G

def green_function_first_derivative(r_source: np.ndarray, r_observer: np.ndarray, k: float) -> np.ndarray:
    r_vector = r_observer - r_source
    r = np.linalg.norm(r_vector, axis=-1)
    dG_dr = (1j * k - 1 / r) * np.exp(1j * k * r) / (4 * np.pi * r ** 2)
    dG_dvector = dG_dr[..., None] * r_vector
    return dG_dvector

def green_function_second_derivative(r_source: np.ndarray, r_observer: np.ndarray, k: float) -> np.ndarray:
    r_vector = r_observer - r_source
    r = np.linalg.norm(r_vector, axis=-1)
    common_coefficient = k * np.exp(1j * k * r) / (4 * np.pi * r ** 2)
    r_vector_outer_product = r_vector[..., :, None] * r_vector[..., None, :]
    diagonal_matrix = np.eye(3)
    second_derivative = -common_coefficient[..., None, None] / r[..., None, None] * (2j / r[..., None, None] + k) * r_vector_outer_product
    second_derivative += common_coefficient[..., None, None] * 1j * diagonal_matrix
    return second_derivative

def normal_to_a_sphere(r_surface: np.ndarray, o_center: np.ndarray, sign: int) -> np.ndarray:
    normal_vector = r_surface - o_center
    normal_vector = normal_vector / np.linalg.norm(normal_vector, axis=-1)[..., np.newaxis] * sign
    return normal_vector

def m_EE(r_source: np.ndarray, r_observer: np.ndarray, k: float, normal_function: Callable) -> np.ndarray:
    dG = green_function_first_derivative(r_source, r_observer, k)
    n = normal_function(r_source)
    if r_source.ndim == 1:
        M = np.array([[-(dG[1] * n[1] + dG[2] * n[2]), dG[1] * n[0], dG[2] * n[0]],
                     [dG[0] * n[1], -(dG[2] * n[2] + dG[0] * n[0]), dG[2] * n[1]],
                     [dG[0] * n[2], dG[1] * n[2], -(dG[0] * n[0] + dG[1] * n[1])]],
            dtype=complex)
    else:
        row0 = np.stack(
            [
                -(dG[..., 1] * n[..., 1] + dG[..., 2] * n[..., 2]),
                dG[..., 1] * n[..., 0],
                dG[..., 2] * n[..., 0],
            ],
            axis=-1,
        )

        row1 = np.stack(
            [
                dG[..., 0] * n[..., 1],
                -(dG[..., 2] * n[..., 2] + dG[..., 0] * n[..., 0]),
                dG[..., 2] * n[..., 1],
            ],
            axis=-1,
        )

        row2 = np.stack(
            [
                dG[..., 0] * n[..., 2],
                dG[..., 1] * n[..., 2],
                -(dG[..., 0] * n[..., 0] + dG[..., 1] * n[..., 1]),
            ],
            axis=-1,
        )

        M = np.stack([row0, row1, row2], axis=-2)
    return M

def m_EH(r_source: np.ndarray, r_observer: np.ndarray, k: float, normal_function: Callable):
    ddG = green_function_second_derivative(r_source, r_observer, k)
    n = normal_function(r_source)
    if r_source.ndim == 1:
        M = np.array(
            [
                [
                    ddG[1, 0] * n[2] - ddG[2, 0] * n[1],
                    ddG[1, 1] * n[2] + ddG[2, 2] * n[2] + ddG[2, 0] * n[0],
                    -(ddG[1, 0] * n[0] + ddG[1, 1] * n[1] + ddG[2, 2] * n[1]),
                ],
                [
                    -(ddG[2, 1] * n[1] + ddG[2, 2] * n[2] + ddG[0, 0] * n[2]),
                    ddG[2, 1] * n[0] - ddG[0, 1] * n[2],
                    ddG[2, 2] * n[0] + ddG[0, 0] * n[0] + ddG[0, 1] * n[1],
                ],
                [
                    ddG[0, 0] * n[1] + ddG[1, 1] * n[1] + ddG[1, 2] * n[2],
                    -(ddG[0, 2] * n[2] + ddG[0, 0] * n[0] + ddG[1, 1] * n[0]),
                    ddG[0, 2] * n[1] - ddG[1, 2] * n[0],
                ],
            ],
            dtype=complex,
        )
    else:
        row0 = np.stack(
            [
                ddG[..., 1, 0] * n[..., 2] - ddG[..., 2, 0] * n[..., 1],
                ddG[..., 1, 1] * n[..., 2] + ddG[..., 2, 2] * n[..., 2] + ddG[..., 2, 0] * n[..., 0],
                -(ddG[..., 1, 0] * n[..., 0] + ddG[..., 1, 1] * n[..., 1] + ddG[..., 2, 2] * n[..., 1]),
            ],
            axis=-1,
        )

        row1 = np.stack(
            [
                -(ddG[..., 2, 1] * n[..., 1] + ddG[..., 2, 2] * n[..., 2] + ddG[..., 0, 0] * n[..., 2]),
                ddG[..., 2, 1] * n[..., 0] - ddG[..., 0, 1] * n[..., 2],
                ddG[..., 2, 2] * n[..., 0] + ddG[..., 0, 0] * n[..., 0] + ddG[..., 0, 1] * n[..., 1],
            ],
            axis=-1,
        )

        row2 = np.stack(
            [
                ddG[..., 0, 0] * n[..., 1] + ddG[..., 1, 1] * n[..., 1] + ddG[..., 1, 2] * n[..., 2],
                -(ddG[..., 0, 2] * n[..., 2] + ddG[..., 0, 0] * n[..., 0] + ddG[..., 1, 1] * n[..., 0]),
                ddG[..., 0, 2] * n[..., 1] - ddG[..., 1, 2] * n[..., 0],
            ],
            axis=-1,
        )

        M = np.stack([row0, row1, row2], axis=-2)
    return M

def m_total(r_source: np.ndarray, r_observer: np.ndarray, k: float, normal_function: Callable, n_index) -> np.ndarray:
    M_EE_matrix = m_EE(r_source, r_observer, k, normal_function)
    M_EH_matrix = m_EH(r_source, r_observer, k, normal_function)
    M_total = np.zeros((*r_source.shape[0:-1], 6, 6), dtype=complex)
    M_total[..., 0:3, 0:3] = M_EE_matrix
    M_total[..., 0:3, 3:6] = 1 / (1j * k * C_LIGHT_SPEED * EPSILON_0_PERMITTIVITY) * M_EH_matrix
    M_total[..., 3:6, 0:3] = - 1 / (1j * k * C_LIGHT_SPEED * MU_0_PERMEABILITY) * M_EH_matrix
    M_total[..., 3:6, 3:6] = M_EE_matrix
    return M_total

def generalized_snells_law(k_vector: np.ndarray,
                           n_forwards: np.ndarray,
                           n_1: float,
                           n_2: float,
                           ) -> np.ndarray:
    cos_theta_incoming = np.clip(np.sum(k_vector * n_forwards, axis=-1), a_min=-1, a_max=1)  # m_rays
    n_tangential = (
            k_vector - cos_theta_incoming[..., np.newaxis] * n_forwards
    )  # m_rays | 3  # This is the vector that is orthogonal to the normal to the surface and lives in the plane spanned by the ray and the normal to the surface (grahm-schmidt process).
    n_tangential_norm = np.linalg.norm(n_tangential, axis=-1)  # m_rays
    if isinstance(n_tangential_norm, float) and n_tangential_norm < 1e-15:
        reflected_direction_vector = n_forwards
    else:
        practically_normal_incidences = n_tangential_norm < 1e-15
        n_tangential[practically_normal_incidences] = (
            np.nan
        )  # This is done so that the normalization does not throw an error.
        n_tangential = normalize_vector(n_tangential)
        sin_theta_outgoing = np.sqrt((n_1 / n_2) ** 2 * (1 - cos_theta_incoming ** 2))  # m_rays
        reflected_direction_vector = (
                n_forwards * stable_sqrt(1 - sin_theta_outgoing[..., np.newaxis] ** 2)
                + n_tangential * sin_theta_outgoing[..., np.newaxis]
        )  # m_rays | 3
        reflected_direction_vector[practically_normal_incidences] = n_forwards[
            practically_normal_incidences
        ]  # For the nans we initiated before, we just want the normal to the surface to be the new direction of the ray
    return reflected_direction_vector  # , n_forwards, n_orthogonal


def generalized_mirror_law(k_vector: np.ndarray, n_forwards: np.ndarray) -> np.ndarray:
    # Notice that this function does not reflect along the normal of the mirror but along the normal projection
    # of the ray on the mirror.
    dot_product = np.sum(k_vector * n_forwards, axis=-1)  # m_rays  # This dot product is written
    # like so because both tensors have the same shape and the dot product is calculated along the last axis.
    # you could also perform this product by transposing the second tensor and then dot multiplying the two tensors,
    # but this it would be cumbersome to do so.
    reflected_direction_vector = (
            k_vector - 2 * dot_product[..., np.newaxis] * n_forwards
    )  # m_rays | 3
    return reflected_direction_vector