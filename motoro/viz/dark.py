import plotly.graph_objects as go

# MODO Primary Color Palettes
modo_primary = {
    # Main palette
    'octarine': '#6f5fe5',
    'red': '#e03f61',
    'orange': '#e59c58',
    'blue': '#467ee4',
    'yellow': '#ffe300',
    'green': '#27ac60',
    'purple': '#a074fb',
    'sand': '#f7c191',
    'light_green': '#6cdd9c',
    'light_purple': '#c5aafd',

    # Alternative palette additions
    'light_blue': '#81b1fe',
    'salmon': '#ee7b6f',
    'orange_red': '#e85f37',
    'dark_red': '#b91e3d'
}

# Ancillary Service Colors (for specific data types)
ancillary_colors = {
    'regup': ['#5fd0e4', '#73d6e7', '#87dceb', '#9be2ee', '#afe8f1', '#c3edf5', '#d7f3f8', '#ebf9fc'],
    'regdown': ['#5743e1', '#6c5be5', '#8172e8', '#968aec', '#aba1f0', '#c0b9f4', '#d5d0f7', '#eae8fb'],
    'rrs': ['#25af60', '#40b974', '#5cc388', '#77cd9c', '#92d7b0', '#ade1c3', '#c8ebd7', '#e4f5eb'],
    'ecrs': ['#ff85a1', '#ff94ad', '#ffa4b9', '#ffb3c4', '#ffc2d0', '#ffd1dc', '#ffe1e8', '#fff0f3'],
    'nsrs': ['#2f80ec', '#4990ee', '#63a0f1', '#7db0f3', '#97c0f5', '#b1cff8', '#cbdffa', '#e5effd'],
    'dam_energy': ['#f2994a', '#f4a661', '#f5b377', '#f7bf8e', '#f8cca5', '#fad9bb', '#fce6d2', '#fdf2e8'],
    'rtm_energy': ['#cc3266', '#d24c79', '#d9658c', '#df7f9f', '#e699b3', '#ecb2c6', '#f2ccd9', '#f9e5ec'],
    'ordc': ['#ffbc00', '#ffc420', '#ffcd40', '#ffd560', '#ffde80', '#ffe69f', '#ffeebf', '#fff7df'],
    'net_revenue': ['#7f70e8', '#8f82eb', '#9f94ee', '#afa6f1', '#bfb7f3', '#cfc9f6', '#dfdbf9', '#efedfc']
}

# Dark mode background colors
dark_backgrounds = {
    'bg0_hard': '#0d0d0d',
    'bg0': '#1a1a1a',
    'bg0_soft': '#262626',
    'bg1': '#333333',
    'bg2': '#404040',
    'bg3': '#4d4d4d',
    'bg4': '#595959',
    'bg_grid': '#2d2d2d',
    'bg_hover': '#2a2a2a'
}

# Light text colors for dark mode
text_colors = {
    'fg0': '#ffffff',
    'fg1': '#f5f5f5',
    'fg2': '#e6e6e6',
    'fg3': '#cccccc',
    'fg4': '#b3b3b3',
    'gray': '#999999'
}

# Create the MODO Dark Layout
modo_dark_layout = go.Layout(
    # Title styling
    title={
        'x': 0.05,
        'font': {
            'color': text_colors['fg1'],
            'size': 20,
            'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
        }
    },

    # Paper and plot backgrounds
    paper_bgcolor=dark_backgrounds['bg0_hard'],
    plot_bgcolor=dark_backgrounds['bg0'],

    # Font defaults
    font={
        'color': text_colors['fg2'],
        'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
        'size': 12
    },

    # Primary color palette for traces
    # Following modo_palette.md order with visualization principles in mind
    colorway=[
        modo_primary['octarine'],      # Strong, distinctive primary
        modo_primary['red'],           # High contrast, attention-grabbing
        modo_primary['orange'],        # Warm, distinguishable
        modo_primary['blue'],          # Cool, professional
        modo_primary['yellow'],        # Bright accent (use sparingly)
        modo_primary['green'],         # Natural, positive
        modo_primary['purple'],        # Secondary distinctive
        modo_primary['sand'],          # Softer option
        modo_primary['light_green'],   # Lighter variant
        modo_primary['light_purple'],  # Lighter variant
        # Additional colors from alt palette for extended needs
        modo_primary['light_blue'],
        modo_primary['salmon'],
        modo_primary['orange_red'],
        modo_primary['dark_red']
    ],

    # Axes styling
    xaxis={
        'automargin': True,
        'gridcolor': dark_backgrounds['bg_grid'],
        'gridwidth': 1,
        'linecolor': dark_backgrounds['bg3'],
        'linewidth': 1,
        'ticks': 'outside',
        'tickcolor': dark_backgrounds['bg3'],
        'tickwidth': 1,
        'tickfont': {
            'color': text_colors['fg3'],
            'size': 11
        },
        'title': {
            'font': {
                'color': text_colors['fg2'],
                'size': 14
            },
            'standoff': 15
        },
        'zerolinecolor': dark_backgrounds['bg4'],
        'zerolinewidth': 1.5,
        'showgrid': True,
        'showline': True,
        'mirror': False
    },

    yaxis={
        'automargin': True,
        'gridcolor': dark_backgrounds['bg_grid'],
        'gridwidth': 1,
        'linecolor': dark_backgrounds['bg3'],
        'linewidth': 1,
        'ticks': 'outside',
        'tickcolor': dark_backgrounds['bg3'],
        'tickwidth': 1,
        'tickfont': {
            'color': text_colors['fg3'],
            'size': 11
        },
        'title': {
            'font': {
                'color': text_colors['fg2'],
                'size': 14
            },
            'standoff': 15
        },
        'zerolinecolor': dark_backgrounds['bg4'],
        'zerolinewidth': 1.5,
        'showgrid': True,
        'showline': True,
        'mirror': False
    },

    # Legend styling
    legend={
        'bgcolor': dark_backgrounds['bg0_soft'],
        'bordercolor': dark_backgrounds['bg3'],
        'borderwidth': 1,
        'font': {
            'color': text_colors['fg2'],
            'size': 11
        },
        'title': {
            'font': {
                'color': text_colors['fg1'],
                'size': 12
            }
        }
    },

    # Hover label styling
    hoverlabel={
        'align': 'left',
        'bgcolor': dark_backgrounds['bg_hover'],
        'bordercolor': modo_primary['octarine'],
        'font': {
            'color': text_colors['fg1'],
            'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            'size': 12
        },
        'namelength': -1
    },

    # Hover mode
    hovermode='closest',

    # Annotations defaults
    annotationdefaults={
        'arrowcolor': text_colors['fg3'],
        'arrowhead': 2,
        'arrowwidth': 1,
        'font': {
            'color': text_colors['fg2'],
            'size': 12
        },
        'showarrow': True
    },

    # Shape defaults
    shapedefaults={
        'line': {'color': text_colors['fg4']},
        'fillcolor': dark_backgrounds['bg2'],
        'opacity': 0.3
    },

    # Color scales using ancillary colors
    colorscale={
        # Sequential colorscale using regulation up colors
        'sequential': [
            [0.0, ancillary_colors['nsrs'][0]],
            [0.125, ancillary_colors['nsrs'][1]],
            [0.25, ancillary_colors['nsrs'][2]],
            [0.375, ancillary_colors['nsrs'][3]],
            [0.5, ancillary_colors['nsrs'][4]],
            [0.625, ancillary_colors['nsrs'][5]],
            [0.75, ancillary_colors['nsrs'][6]],
            [0.875, ancillary_colors['nsrs'][7]],
            [1.0, ancillary_colors['ordc'][0]]
        ],
        # Diverging colorscale from blue through dark to red
        'diverging': [
            [0.0, modo_primary['blue']],
            [0.1, ancillary_colors['nsrs'][2]],
            [0.3, ancillary_colors['nsrs'][5]],
            [0.45, dark_backgrounds['bg3']],
            [0.5, dark_backgrounds['bg4']],
            [0.55, dark_backgrounds['bg3']],
            [0.7, ancillary_colors['ecrs'][2]],
            [0.9, ancillary_colors['rtm_energy'][1]],
            [1.0, modo_primary['red']]
        ],
        # Sequential minus using net revenue colors
        'sequentialminus': [
            [0.0, dark_backgrounds['bg1']],
            [0.125, ancillary_colors['net_revenue'][7]],
            [0.25, ancillary_colors['net_revenue'][6]],
            [0.375, ancillary_colors['net_revenue'][5]],
            [0.5, ancillary_colors['net_revenue'][4]],
            [0.625, ancillary_colors['net_revenue'][3]],
            [0.75, ancillary_colors['net_revenue'][2]],
            [0.875, ancillary_colors['net_revenue'][1]],
            [1.0, ancillary_colors['net_revenue'][0]]
        ]
    },

    # 3D scene configuration
    scene={
        'xaxis': {
            'backgroundcolor': dark_backgrounds['bg0'],
            'gridcolor': dark_backgrounds['bg_grid'],
            'gridwidth': 1,
            'linecolor': dark_backgrounds['bg3'],
            'showbackground': True,
            'ticks': 'outside',
            'tickcolor': dark_backgrounds['bg3'],
            'tickfont': {'color': text_colors['fg3']},
            'title': {'font': {'color': text_colors['fg2']}},
            'zerolinecolor': dark_backgrounds['bg4']
        },
        'yaxis': {
            'backgroundcolor': dark_backgrounds['bg0'],
            'gridcolor': dark_backgrounds['bg_grid'],
            'gridwidth': 1,
            'linecolor': dark_backgrounds['bg3'],
            'showbackground': True,
            'ticks': 'outside',
            'tickcolor': dark_backgrounds['bg3'],
            'tickfont': {'color': text_colors['fg3']},
            'title': {'font': {'color': text_colors['fg2']}},
            'zerolinecolor': dark_backgrounds['bg4']
        },
        'zaxis': {
            'backgroundcolor': dark_backgrounds['bg0'],
            'gridcolor': dark_backgrounds['bg_grid'],
            'gridwidth': 1,
            'linecolor': dark_backgrounds['bg3'],
            'showbackground': True,
            'ticks': 'outside',
            'tickcolor': dark_backgrounds['bg3'],
            'tickfont': {'color': text_colors['fg3']},
            'title': {'font': {'color': text_colors['fg2']}},
            'zerolinecolor': dark_backgrounds['bg4']
        },
        'camera': {
            'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}
        }
    },

    # Polar configuration
    polar={
        'bgcolor': dark_backgrounds['bg0'],
        'angularaxis': {
            'gridcolor': dark_backgrounds['bg_grid'],
            'linecolor': dark_backgrounds['bg3'],
            'ticks': 'outside',
            'tickcolor': dark_backgrounds['bg3'],
            'tickfont': {'color': text_colors['fg3']}
        },
        'radialaxis': {
            'gridcolor': dark_backgrounds['bg_grid'],
            'linecolor': dark_backgrounds['bg3'],
            'ticks': 'outside',
            'tickcolor': dark_backgrounds['bg3'],
            'tickfont': {'color': text_colors['fg3']}
        }
    },

    # Ternary configuration
    ternary={
        'bgcolor': dark_backgrounds['bg0'],
        'aaxis': {
            'gridcolor': dark_backgrounds['bg_grid'],
            'linecolor': dark_backgrounds['bg3'],
            'ticks': 'outside',
            'tickcolor': dark_backgrounds['bg3'],
            'tickfont': {'color': text_colors['fg3']},
            'title': {'font': {'color': text_colors['fg2']}}
        },
        'baxis': {
            'gridcolor': dark_backgrounds['bg_grid'],
            'linecolor': dark_backgrounds['bg3'],
            'ticks': 'outside',
            'tickcolor': dark_backgrounds['bg3'],
            'tickfont': {'color': text_colors['fg3']},
            'title': {'font': {'color': text_colors['fg2']}}
        },
        'caxis': {
            'gridcolor': dark_backgrounds['bg_grid'],
            'linecolor': dark_backgrounds['bg3'],
            'ticks': 'outside',
            'tickcolor': dark_backgrounds['bg3'],
            'tickfont': {'color': text_colors['fg3']},
            'title': {'font': {'color': text_colors['fg2']}}
        }
    },

    # Geo configuration
    geo={
        'bgcolor': dark_backgrounds['bg0'],
        'lakecolor': dark_backgrounds['bg0_hard'],
        'landcolor': dark_backgrounds['bg1'],
        'showlakes': True,
        'showland': True,
        'subunitcolor': dark_backgrounds['bg3'],
        'coastlinecolor': dark_backgrounds['bg4'],
        'projection': {'type': 'natural earth'},
        'showframe': True,
        'framecolor': dark_backgrounds['bg3']
    },

    # Mapbox style
    mapbox={
        'style': 'dark',
        # 'accesstoken': ''  # Add your mapbox token if using mapbox plots
    },

    # Color axis
    coloraxis={
        'colorbar': {
            'outlinewidth': 0,
            'ticks': 'outside',
            'tickcolor': dark_backgrounds['bg3'],
            'tickfont': {'color': text_colors['fg3']},
            'title': {'font': {'color': text_colors['fg2']}}
        }
    },

    # Selection styling
    # selectdirection='diagonal',
    selectdirection='d',

    # Drag mode
    dragmode='zoom',

    # Strict number formatting
    autotypenumbers='strict',

    # Uniform text styling
    uniformtext={
        'minsize': 8,
        'mode': 'hide'
    },

    # Margin
    margin={
        'l': 80,
        'r': 80,
        't': 100,
        'b': 80,
        'pad': 4
    },

    # Transition settings
    transition={
        'duration': 500,
        'easing': 'cubic-in-out'
    }
)

# Example usage:
# fig = go.Figure(data=[...], layout=modo_dark_layout)
# Or update an existing figure:
# fig.update_layout(modo_dark_layout)

# For specific ancillary service visualizations, you can use:
# fig.update_traces(marker_color=ancillary_colors['regup'][0])  # For regulation up data
# fig.update_traces(marker_color=ancillary_colors['dam_energy'][0])  # For DAM energy data
