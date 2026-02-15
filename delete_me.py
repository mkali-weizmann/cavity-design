import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

from ipywidgets import Layout, FloatSlider, Checkbox, widgets, IntSlider

from simple_analysis_scripts.potential_analysis.analyze_potential import *

import io
from PIL import Image
import win32clipboard

def copy_figure_to_clipboard(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    image = Image.open(buf)
    output = io.BytesIO()
    image.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]  # BMP header removed

    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
    win32clipboard.CloseClipboard()




def f(
        n_actual_aspheric,
        n_rays,
        copy_image,
        unconcentricity,
        phi_max,
        defocus,
        back_focal_length_aspheric,
        T_c_aspheric,
        n_design_aspheric,
        n_design_spherical,
        n_actual_spherical,
        T_c_spherical,
        f_spherical,
        diameter,
):
    optical_system = generate_two_lenses_optical_system(defocus=defocus, back_focal_length_aspheric=back_focal_length_aspheric, T_c_aspheric=T_c_aspheric, n_design_aspheric=n_design_aspheric, n_actual_aspheric=n_actual_aspheric, n_design_spherical=n_design_spherical, n_actual_spherical=n_actual_spherical, T_c_spherical=T_c_spherical, f_spherical=f_spherical, diameter=diameter, )
    rays_0 = initialize_rays(defocus=defocus, n_rays=n_rays, phi_max=phi_max)
    results_dict = analyze_potential(optical_system=optical_system, rays_0=rays_0, unconcentricity=unconcentricity, print_tests=True, )
    fig, ax = plot_results(results_dict=results_dict, far_away_plane=True, unconcentricity=unconcentricity, potential_x_axis_angles=False, )

    if copy_image:
        copy_figure_to_clipboard(fig)

# rest of parameters
n_rays = 50
unconcentricity = 5e-3  # np.float64(0.007610344827586207)  # ,  np.float64(0.007268965517241379)
phi_max = 0.3
defocus = 0.0029887489470528557
back_focal_length_aspheric = 20e-3
T_c_aspheric = 4.35e-3
n_design_aspheric = 1.45
n_actual_aspheric = 1.45
n_design_spherical = 1.45
n_actual_spherical = 1.45
T_c_spherical = 4.35e-3
f_spherical = 100e-3
diameter = 12.7e-3

widgets.interact(
    f,
    n_actual_aspheric=FloatSlider(value=n_actual_aspheric, min=1.0, max=2.0, step=0.0001, description='n actual', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
    n_rays=IntSlider(value=n_rays, min=2, max=2000, step=1, description='n rays', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
    copy_image=Checkbox(value=False, description='Copy image', style={'description_width': 'initial'}),
    unconcentricity=FloatSlider(value=unconcentricity, min=0.0, max=0.02, step=1e-5, description='unconcentricity', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
    phi_max=FloatSlider(value=phi_max, min=0.0, max=1.0, step=0.001, description='phi max', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
    defocus=FloatSlider(value=defocus, min=-0.01, max=0.01, step=1e-6, description='defocus', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
    back_focal_length_aspheric=FloatSlider(value=back_focal_length_aspheric, min=1e-3, max=0.1, step=1e-4, description='back focal length (aspheric)', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
    T_c_aspheric=FloatSlider(value=T_c_aspheric, min=0.0, max=0.02, step=1e-5, description='T_c aspheric', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
    n_design_aspheric=FloatSlider(value=n_design_aspheric, min=1.0, max=2.0, step=0.0001, description='n design (aspheric)', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
    n_design_spherical=FloatSlider(value=n_design_spherical, min=1.0, max=2.0, step=0.0001, description='n design (spherical)', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
    n_actual_spherical=FloatSlider(value=n_actual_spherical, min=1.0, max=2.0, step=0.0001, description='n actual (spherical)', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
    T_c_spherical=FloatSlider(value=T_c_spherical, min=0.0, max=0.02, step=1e-5, description='T_c spherical', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
    f_spherical=FloatSlider(value=f_spherical, min=1e-3, max=0.5, step=1e-4, description='f spherical', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
    diameter=FloatSlider(value=diameter, min=1e-3, max=0.05, step=1e-4, description='diameter', layout=Layout(width='1500px'), style={'description_width': 'initial'}),
)
