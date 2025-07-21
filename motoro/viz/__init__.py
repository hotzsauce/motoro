"""
To use the plotly templates, use the following boilerplate code

pd.options.plotting.backend = "plotly"

import plotly.graph_objects as go
import plotly.io as io

from motoro.viz import dark, light

pio.templates["modo-dark"] = go.layout.Template(layout=dark)
pio.templates["modo-light"] = go.layout.Template(layout=light)

# pio.templates.default = "modo-light" # or "modo-dark"
"""
from motoro.viz.dark import modo_dark_layout as dark
from motoro.viz.light import modo_light_layout as light
