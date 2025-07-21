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

# Darker variants for better contrast on light backgrounds
modo_dark_variants = {
    'octarine_dark': '#5a4cc7',
    'blue_dark': '#2f5cc7',
    'green_dark': '#1d8549',
    'purple_dark': '#8659e0',
    'yellow_dark': '#d4b800',  # Darker yellow for better visibility
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

# Light mode background colors
light_backgrounds = {
    'bg0_hard': '#ffffff',
    'bg0': '#fcfcfc',
    'bg0_soft': '#f8f8f8',
    'bg1': '#f3f3f3',
    'bg2': '#ededed',
    'bg3': '#e6e6e6',
    'bg4': '#d9d9d9',
    'bg_grid': '#f0f0f0',
    'bg_hover': '#f5f5f5'
}

# Dark text colors for light mode
text_colors = {
    'fg0': '#000000',
    'fg1': '#1a1a1a',
    'fg2': '#333333',
    'fg3': '#4d4d4d',
    'fg4': '#666666',
    'gray': '#808080'
}

# Create the MODO Light Layout
modo_light_layout = go.Layout(
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
    paper_bgcolor=light_backgrounds['bg0_hard'],
    plot_bgcolor=light_backgrounds['bg0'],

    # Font defaults
    font={
        'color': text_colors['fg2'],
        'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
        'size': 12
    },

    # Primary color palette for traces
    # Following modo_palette.md order with adjustments for light mode
    colorway=[
        modo_dark_variants['octarine_dark'],  # Darker for better contrast
        modo_primary['red'],                  # Already has good contrast
        modo_primary['orange'],               # Good contrast
        modo_dark_variants['blue_dark'],      # Darker blue for visibility
        modo_dark_variants['yellow_dark'],    # Much darker yellow
        modo_dark_variants['green_dark'],     # Darker green
        modo_dark_variants['purple_dark'],    # Darker purple
        modo_primary['dark_red'],             # Using dark_red instead of sand
        modo_primary['green'],                # Original green as alternative
        modo_primary['purple'],               # Original purple as alternative
        # Additional colors
        modo_primary['blue'],
        modo_primary['orange_red'],
        modo_primary['salmon'],
        modo_primary['octarine']
    ],

    # Axes styling
    xaxis={
        'automargin': True,
        'gridcolor': light_backgrounds['bg_grid'],
        'gridwidth': 1,
        'linecolor': light_backgrounds['bg4'],
        'linewidth': 1,
        'ticks': 'outside',
        'tickcolor': light_backgrounds['bg4'],
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
        'zerolinecolor': light_backgrounds['bg4'],
        'zerolinewidth': 1.5,
        'showgrid': True,
        'showline': True,
        'mirror': False
    },

    yaxis={
        'automargin': True,
        'gridcolor': light_backgrounds['bg_grid'],
        'gridwidth': 1,
        'linecolor': light_backgrounds['bg4'],
        'linewidth': 1,
        'ticks': 'outside',
        'tickcolor': light_backgrounds['bg4'],
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
        'zerolinecolor': light_backgrounds['bg4'],
        'zerolinewidth': 1.5,
        'showgrid': True,
        'showline': True,
        'mirror': False
    },

    # Legend styling
    legend={
        'bgcolor': light_backgrounds['bg0_soft'],
        'bordercolor': light_backgrounds['bg3'],
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
        'bgcolor': light_backgrounds['bg_hover'],
        'bordercolor': modo_dark_variants['octarine_dark'],
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
        'fillcolor': light_backgrounds['bg2'],
        'opacity': 0.3
    },

    # Color scales using ancillary colors
    colorscale={
        # Sequential colorscale - darker start for light backgrounds
        'sequential': [
            [0.0, ancillary_colors['nsrs'][0]],
            [0.125, ancillary_colors['nsrs'][1]],
            [0.25, ancillary_colors['nsrs'][2]],
            [0.375, ancillary_colors['nsrs'][3]],
            [0.5, ancillary_colors['nsrs'][4]],
            [0.625, ancillary_colors['dam_energy'][1]],
            [0.75, ancillary_colors['dam_energy'][0]],
            [0.875, modo_primary['orange_red']],
            [1.0, modo_primary['dark_red']]
        ],
        # Diverging colorscale from blue through light gray to red
        'diverging': [
            [0.0, modo_dark_variants['blue_dark']],
            [0.1, modo_primary['blue']],
            [0.3, ancillary_colors['nsrs'][4]],
            [0.45, light_backgrounds['bg3']],
            [0.5, light_backgrounds['bg2']],
            [0.55, light_backgrounds['bg3']],
            [0.7, ancillary_colors['ecrs'][0]],
            [0.9, modo_primary['red']],
            [1.0, modo_primary['dark_red']]
        ],
        # Sequential minus using regdown colors (darker purples)
        'sequentialminus': [
            [0.0, light_backgrounds['bg1']],
            [0.125, ancillary_colors['regdown'][6]],
            [0.25, ancillary_colors['regdown'][5]],
            [0.375, ancillary_colors['regdown'][4]],
            [0.5, ancillary_colors['regdown'][3]],
            [0.625, ancillary_colors['regdown'][2]],
            [0.75, ancillary_colors['regdown'][1]],
            [0.875, ancillary_colors['regdown'][0]],
            [1.0, modo_dark_variants['purple_dark']]
        ]
    },

    # 3D scene configuration
    scene={
        'xaxis': {
            'backgroundcolor': light_backgrounds['bg0'],
            'gridcolor': light_backgrounds['bg_grid'],
            'gridwidth': 1,
            'linecolor': light_backgrounds['bg4'],
            'showbackground': True,
            'ticks': 'outside',
            'tickcolor': light_backgrounds['bg4'],
            'tickfont': {'color': text_colors['fg3']},
            'title': {'font': {'color': text_colors['fg2']}},
            'zerolinecolor': light_backgrounds['bg4']
        },
        'yaxis': {
            'backgroundcolor': light_backgrounds['bg0'],
            'gridcolor': light_backgrounds['bg_grid'],
            'gridwidth': 1,
            'linecolor': light_backgrounds['bg4'],
            'showbackground': True,
            'ticks': 'outside',
            'tickcolor': light_backgrounds['bg4'],
            'tickfont': {'color': text_colors['fg3']},
            'title': {'font': {'color': text_colors['fg2']}},
            'zerolinecolor': light_backgrounds['bg4']
        },
        'zaxis': {
            'backgroundcolor': light_backgrounds['bg0'],
            'gridcolor': light_backgrounds['bg_grid'],
            'gridwidth': 1,
            'linecolor': light_backgrounds['bg4'],
            'showbackground': True,
            'ticks': 'outside',
            'tickcolor': light_backgrounds['bg4'],
            'tickfont': {'color': text_colors['fg3']},
            'title': {'font': {'color': text_colors['fg2']}},
            'zerolinecolor': light_backgrounds['bg4']
        },
        'camera': {
            'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}
        }
    },

    # Polar configuration
    polar={
        'bgcolor': light_backgrounds['bg0'],
        'angularaxis': {
            'gridcolor': light_backgrounds['bg_grid'],
            'linecolor': light_backgrounds['bg4'],
            'ticks': 'outside',
            'tickcolor': light_backgrounds['bg4'],
            'tickfont': {'color': text_colors['fg3']}
        },
        'radialaxis': {
            'gridcolor': light_backgrounds['bg_grid'],
            'linecolor': light_backgrounds['bg4'],
            'ticks': 'outside',
            'tickcolor': light_backgrounds['bg4'],
            'tickfont': {'color': text_colors['fg3']}
        }
    },

    # Ternary configuration
    ternary={
        'bgcolor': light_backgrounds['bg0'],
        'aaxis': {
            'gridcolor': light_backgrounds['bg_grid'],
            'linecolor': light_backgrounds['bg4'],
            'ticks': 'outside',
            'tickcolor': light_backgrounds['bg4'],
            'tickfont': {'color': text_colors['fg3']},
            'title': {'font': {'color': text_colors['fg2']}}
        },
        'baxis': {
            'gridcolor': light_backgrounds['bg_grid'],
            'linecolor': light_backgrounds['bg4'],
            'ticks': 'outside',
            'tickcolor': light_backgrounds['bg4'],
            'tickfont': {'color': text_colors['fg3']},
            'title': {'font': {'color': text_colors['fg2']}}
        },
        'caxis': {
            'gridcolor': light_backgrounds['bg_grid'],
            'linecolor': light_backgrounds['bg4'],
            'ticks': 'outside',
            'tickcolor': light_backgrounds['bg4'],
            'tickfont': {'color': text_colors['fg3']},
            'title': {'font': {'color': text_colors['fg2']}}
        }
    },

    # Geo configuration
    geo={
        'bgcolor': light_backgrounds['bg0'],
        'lakecolor': light_backgrounds['bg_hover'],
        'landcolor': light_backgrounds['bg1'],
        'showlakes': True,
        'showland': True,
        'subunitcolor': light_backgrounds['bg3'],
        'coastlinecolor': light_backgrounds['bg4'],
        'projection': {'type': 'natural earth'},
        'showframe': True,
        'framecolor': light_backgrounds['bg4']
    },

    # Mapbox style
    mapbox={
        'style': 'light',
        # 'accesstoken': ''  # Add your mapbox token if using mapbox plots
    },

    # Color axis
    coloraxis={
        'colorbar': {
            'outlinewidth': 0,
            'ticks': 'outside',
            'tickcolor': light_backgrounds['bg4'],
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
# fig = go.Figure(data=[...], layout=modo_light_layout)
# Or update an existing figure:
# fig.update_layout(modo_light_layout)

# For specific ancillary service visualizations, you can use:
# fig.update_traces(marker_color=ancillary_colors['regup'][0])  # For regulation up data
# fig.update_traces(marker_color=ancillary_colors['dam_energy'][0])  # For DAM energy data
