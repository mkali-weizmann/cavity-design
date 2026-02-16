from simple_analysis_scripts.potential_analysis.analyze_potential import *
import sys
from pathlib import Path

project_root = Path.cwd().parent.parent
sys.path.append(str(project_root))

from ipywidgets import Layout, FloatSlider, Checkbox, Text, widgets, IntSlider, Dropdown

from simple_analysis_scripts.potential_analysis.analyze_potential import *

import io
import matplotlib.pyplot as plt
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
# %%
widget_convenient_exponent()


